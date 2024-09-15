import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
import time
from loguru import logger
import gc
from copy import deepcopy

from models.modules import SPVCNN
from utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project import back_project
from ops.generate_grids import generate_grid
from skimage import measure
import trimesh
from datasets.visualization import visualize_mesh
from models.modules import Linear4xTrans, Panoptic_Feat_Fusion
from models.occupancy_initialization import Occupancy_Initialization, Back_Project, get_img_feats
from models.mask3dformer import MultiScaleMaskedTransformerDecoder, panoptic_post
from models.criterion import SetCriterion
from models.matcher import HungarianMatcher

class NeuConNet(nn.Module):
    def __init__(self, cfg):
        super(NeuConNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        ch_in = [80 * alpha, 96 + 40 * alpha + 2, 48 + 24 * alpha + 2, 24 + 24 + 2]  # default: [81, 139, 75, 51] (Features after fusion)
        channels = [96, 48, 24]  # (Features converted from fusion to prediction)
        ch_initialization = [80, 40, 24]

        ch_initialization_down = 32
        n_views = 9
        panoptic_channels = 48
        GRU_channels = [x + y for x, y in zip(channels, ch_initialization)]

        # initialization
        self.back_projection = nn.ModuleList()
        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg, ch_in=GRU_channels, ch_voxel=channels)
        # sparse conv
        self.sp_convs = nn.ModuleList()
        # MLPs that predict tsdf and occupancy.
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        self.panoptic_preds = nn.ModuleList()

        # initialization
        self.initialization = Occupancy_Initialization(ch_initialization, ch_initialization_down, n_views)

        # panoptic_feat_fusion
        self.panoptic_feat_fusion = Panoptic_Feat_Fusion(channels[2], panoptic_channels, ch_initialization)

        # panoptic
        num_classes = 20
        self.dec_layers = 6
        self.panoptic = MultiScaleMaskedTransformerDecoder(mask_classification=True,
                                                           num_classes=num_classes,
                                                           hidden_dim=panoptic_channels,
                                                           num_queries=80,
                                                           nheads=8,
                                                           dim_feedforward=4 * panoptic_channels,
                                                           dec_layers=self.dec_layers,
                                                           pre_norm=False,
                                                           mask_dim=panoptic_channels,
                                                           )

        # criterion
        class_weight, mask_weight, dice_weight = 0.2, 0.8, 0.8
        no_object_weight = 0.1

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        weight_dict_temp = deepcopy(weight_dict)
        for i in range(self.dec_layers):
            weight_dict.update({k + f"_{i}": v for k, v in weight_dict_temp.items()})
        del weight_dict_temp

        losses = ["labels", "masks"]

        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,  # weight_dict
            eos_coef=no_object_weight,
            losses=losses,
        )

        for i in range(len(cfg.THRESHOLDS)):
            # back projection
            self.back_projection.append(Back_Project(ch_initialization[i]))

            self.sp_convs.append(
                SPVCNN(num_classes=1, in_channels=ch_in[i],
                       pres=1,
                       cr=1 / 2 ** i,
                       vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
                       dropout=self.cfg.SPARSEREG.DROPOUT)
            )

            self.tsdf_preds.append(Linear4xTrans(channels[i], 1))
            self.occ_preds.append(Linear4xTrans(channels[i], 1))
            self.panoptic_preds.append(Linear4xTrans(GRU_channels[i], panoptic_channels))

    def get_target(self, coords, inputs, scale):
        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, occ_target

    def get_target_init(self, coords, inputs, scale):
        with torch.no_grad():
            occ_target = inputs['occ_list'][scale]
            tsdf_target = inputs['tsdf_list'][scale]

            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]

            tsdf_target = torch.abs(tsdf_target)
            tsdf_target = 1 - tsdf_target
            tsdf_target = torch.clamp(tsdf_target, min=0, max=1)

            return tsdf_target, occ_target

    def get_xyzrgb_targets(self, coords, inputs, scale):
        with torch.no_grad():
            rgb_target = inputs['rgb_list'][scale]

            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            rgb_target = rgb_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3], :]
            xyzrgb_target = torch.concat([coords[:, 1:], rgb_target], dim=-1)

            return xyzrgb_target

    def get_panoptic_targets(self, coords, inputs, scale, bs):
        with torch.no_grad():
            semantic_target = inputs['semantic_list'][scale]
            instance_target = inputs['instance_list'][scale]

            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            semantic_target = semantic_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            instance_target = instance_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]

            # aa, bb = torch.max(semantic_target), torch.min(semantic_target)
            # cc, dd = torch.max(instance_target), torch.min(instance_target)

            panoptic_targets = []
            for batch in range(bs):
                panoptic_targets.append({})
                batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)

                semantic_label = semantic_target[batch_ind]  # [num_voxel, ] class_id
                instance_label = instance_target[batch_ind]  # [num_voxel, ] instance_id

                labels = []
                masks = []
                unique_ids = torch.unique(instance_label)
                for unique_id in unique_ids:
                    unique_indice = torch.where(instance_label == unique_id)[0]

                    labels.append(torch.argmax(torch.bincount(semantic_label[unique_indice].to(torch.int)))) # In case instance labels and category labels are not equal, select the most
                    masks.append((instance_label == unique_id).unsqueeze(0))

                panoptic_targets[-1]['labels'] = torch.tensor(labels).to(instance_target.device)  # (num_instance, )
                panoptic_targets[-1]['masks'] = torch.cat(masks, dim=0)  # (num_instance, num_voxel)

                del labels, masks

            return panoptic_targets

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]  # 1:x, 2:y, 3:z
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def erode(self, voxels, kernel_size=3):
        """ Erosion operation on the voxels """
        padding = kernel_size // 2
        struct_elem = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=voxels.device)
        eroded = F.conv3d(voxels.float().unsqueeze(0).unsqueeze(0), struct_elem, padding=padding, stride=1)
        return (eroded == struct_elem.numel()).type(torch.bool).squeeze()

    def dilate(self, voxels, kernel_size=3):
        """ Dilation operation on voxels """
        padding = kernel_size // 2
        struct_elem = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=voxels.device)
        dilated = F.conv3d(voxels.float().unsqueeze(0).unsqueeze(0), struct_elem, padding=padding, stride=1)
        return (dilated >= 1).type(torch.bool).squeeze()

    def forward(self, features, features_backbone2d_occ_pano, inputs, outputs, only_train_init=False, only_train_occ=False, init_overlap_count=0):
        bs = features[0][0].shape[0]  # features: list9: list3: 1/4-[1, 24, 120, 160], 1/8-[1, 40, 60, 80], 1/16-[1, 80, 30, 40]
        pre_feat = None
        pre_coords = None
        loss_dict = {}

        # TODO ---------------------------------------------------------------------------------------------------------
        # TODO --------------------------------------   Depth Prior Estimation   ---------------------------------------
        # TODO ---------------------------------------------------------------------------------------------------------
        t_init = time.time()
        init_stage = 1  # Initialization is performed at which stage, default: 1
        interval = 2 ** (self.n_scales - init_stage)
        scale = self.n_scales - init_stage
        min_view_number = 2
        occ_init_thresd = 0.3

        coords, shape_init = generate_grid(self.cfg.N_VOX, interval)  # generate new coords
        up_coords = []
        for b in range(bs):
            up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
        up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()  # [num_voxels, bxyz]
        up_coords = up_coords.to(torch.int32)

        KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()

        init_output = self.initialization(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, features, KRcam, shape_init, init_stage, min_view_number)

        if init_output == None:
            loss_dict.update({f'occupancy_initialization_loss': torch.Tensor([0.0]).cuda()[0] * features[0][0].sum()})
            logger.warning('no valid points in initialization')
            outputs['init_overlap_count'] = init_overlap_count
            return outputs, loss_dict
        occ_init, coord_init, count_init = init_output

        occ_init_selected = occ_init.sigmoid().squeeze(-1) > occ_init_thresd

        # for batch in range(bs):
        #     batch_ind_init = torch.nonzero(coord_init[:, 0] == batch).squeeze(1)
        #     print('initialization points in >2 views: ', len(batch_ind_init), 'occupied: ', occ_init_selected[batch_ind_init].sum(), 'batch: ', batch)

        if only_train_init:
            tsdf_init_target, occ_init_target = self.get_target_init(coord_init, inputs, scale)
            loss_init = []
            for batch in range(bs):
                loss_init.append([])
                batch_ind = torch.nonzero(coord_init[:, 0] == batch).squeeze(1)

                # visualize
                visualize_mesh(coord_init[batch_ind][:, 1:].detach().cpu().numpy(), tsdf_init_target[batch_ind].detach().cpu().numpy(), type='tsdf')
                visualize_mesh(coord_init[batch_ind][:, 1:].detach().cpu().numpy(), occ_init_selected[batch_ind].long().detach().cpu().numpy(), type='tsdf')

                if occ_init_target[batch_ind] is not None:
                    occ_init_overlap = occ_overlap(occ_init[batch_ind].sigmoid() > occ_init_thresd, tsdf_init_target[batch_ind] > 0)
                    init_overlap_count += occ_init_overlap
                    outputs['init_overlap_count'] = init_overlap_count
                    print('occ_init_overlap: ', occ_init_overlap, 'occ_init_thresd: ', occ_init_thresd)
                    # print('occ_init_selected_batch: ', occ_init_selected[batch_ind].sum(), '/', len(occ_init_selected[batch_ind]))
                    loss_init[-1] = self.compute_loss_init(coord_init[:, 1:][batch_ind], occ_init[batch_ind], tsdf_init_target[batch_ind], occ_init_target[batch_ind])  # loss per voxel
                else:
                    loss_init[-1] = torch.Tensor(np.array([0]))[0]
            loss_init = sum(loss_init) / len(loss_init)  # loss per bs per voxel
            loss_dict.update({f'occupancy_initialization_loss': loss_init})

            del coords, init_output, occ_init, occ_init_selected, batch_ind, occ_init_target

            return outputs, loss_dict

        '-- downsample to coarse ---'
        coord_init_selected = []
        for batch in range(bs):
            batch_ind_up = torch.nonzero(up_coords[:, 0] == batch).squeeze(1)
            batch_ind_init = torch.nonzero(coord_init[:, 0] == batch).squeeze(1)

            valid_volume_init = torch.zeros(shape_init, dtype=torch.bool).to(count_init.device)
            valid_volume_init[count_init[batch_ind_up].view(shape_init) >= min_view_number] = occ_init_selected[batch_ind_init]

            valid_volume_init = F.max_pool3d(valid_volume_init.unsqueeze(0).float(), 2 ** init_stage).squeeze(0)
            # valid_volume_init = -F.max_pool3d(-valid_volume_init.unsqueeze(0).float(), 2 ** init_stage).squeeze(0)
            valid_volume_init = self.erode(valid_volume_init, kernel_size=3)
            valid_volume_init = self.dilate(valid_volume_init, kernel_size=3)
            valid_volume_init = self.dilate(valid_volume_init, kernel_size=3)

            valid_volume_init = torch.nonzero(valid_volume_init).squeeze(1)
            bs_index_init = torch.ones(len(valid_volume_init)).to(valid_volume_init.dtype).to(valid_volume_init.device) * batch
            valid_volume_init = torch.concat([bs_index_init.view(-1, 1), valid_volume_init * 4], dim=1)

            coord_init_selected.append(valid_volume_init)

        coord_init_selected = torch.concat(coord_init_selected, dim=0)

        # print("occupancy initialization finished ", (time.time() - t_init) * 1000, "ms", len(count_init), '->', len(occ_init), ' -> ', occ_init_selected.sum())

        # # visualization
        # xyzrgb_target_init = self.get_xyzrgb_targets(coord_init, inputs, scale)
        # occ_init_target = self.get_target_init(coord_init, inputs, scale)
        # self.occupancy_target_visualization(bs, coord_init, xyzrgb_target_init, occ_init_selected, occ_init_target, only_init=True)
        #
        # tsdf2mesh(coord_init[:, 1:] / (2 ** (2 - init_stage)), occ_init_target, shape_init, 'results/tsdf_target.ply')

        # prepare for coarse stage
        count_init = count_init >= min_view_number
        count_init_temp = []
        for b in range(bs):
            batch_ind = torch.nonzero(up_coords[:, 0] == batch).squeeze(1)
            count_init_temp.append(F.max_pool3d(
                count_init[batch_ind].view(shape_init[0], shape_init[1], shape_init[2]).unsqueeze(0).float(),
                2 ** init_stage).bool().squeeze(0))
            count_init_temp[-1] = count_init_temp[-1].view(-1)
        count_init = torch.concat(count_init_temp, dim=0)

        shape_init = tuple(int(shape_init_ / (2 ** init_stage)) for shape_init_ in shape_init)

        del coords, init_output, occ_init, occ_init_selected, batch_ind_up, batch_ind_init, valid_volume_init, bs_index_init, count_init_temp

        # TODO ---------------------------------------------------------------------------------------------------------
        # TODO --------------------------------------   Surface Reconstruction   ---------------------------------------
        # TODO ---------------------------------------------------------------------------------------------------------
        panoptic_voxel_feats, panoptic_coords = [], []
        for i in range(self.cfg.N_LAYER):  # coarse-to-fine
            interval = 2 ** (self.n_scales - i)
            scale = self.n_scales - i

            if i == 0:  # stage 1
                shape = shape_init
                up_coords = coord_init_selected.contiguous()  # [num_voxels, bxyz]
                up_coords = up_coords.to(torch.int32)
                min_view_number = 2
            else:  # Stages 2 and 3
                # ----upsample coords----
                # up_feat, up_panoptic_masks, up_coords = self.upsample(pre_feat, pre_panoptic_masks, pre_coords, interval)  # up_coords: world coordinate of fragment
                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)  # up_coords: world coordinate of fragment
                min_view_number = 0

            # ----back project----
            feats = torch.stack([feat[scale] for feat in features_backbone2d_occ_pano])
            KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()

            # Unproject the image fetures to form a 3D (sparse) feature volume
            # feats = self.back_projection[i].transfer(feats)
            project_output = self.back_projection[i](up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feats, KRcam, min_view_number)
            if project_output == None:
                loss_dict.update({f'tsdf_occ_loss_{i}': torch.Tensor([0.0]).cuda()[0] * feats.sum()})
                logger.warning('no valid points in back_projection: scale {}'.format(i))
                return outputs, loss_dict
            volume, up_coords, reference_points_cam, mask, count = project_output

            # ----concat feature from last stage----
            if i != 0:
                feat = torch.cat([volume, up_feat[count >= min_view_number]], dim=1)  # concat this_layer 和 last_layer
            else:
                feat = volume

            if not self.cfg.FUSION.FUSION_ON:
                tsdf_target, occ_target = self.get_target(up_coords, inputs, scale)

            # ----convert to aligned camera coordinate----
            # with torch.cuda.amp.autocast(enabled=False):
            r_coords = up_coords.detach().clone().float()
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1)
                coords_batch = up_coords[batch_ind][:, 1:].float()
                coords_batch = coords_batch * self.cfg.VOXEL_SIZE + inputs['vol_origin_partial'][b].float()  # fragment_to_world
                coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)
                # world_to_aligned_camera
                coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(1, 0).contiguous()
                r_coords[batch_ind, 1:] = coords_batch  # 每个voxel在相机坐标系下的齐次坐标

            # batch index is in the last position
            r_coords = r_coords[:, [1, 2, 3, 0]]  # bxyz to xyzb

            # ----sparse conv 3d backbone----
            point_feat = PointTensor(feat, r_coords)  # feat: (voxel_num, dim), r_coords: (voxel_num, x_y_z_bs)
            feat = self.sp_convs[i](point_feat)  # feat: (voxel_num, new_channels)

            feat_all = torch.cat([feat, volume], dim=-1)
            # ----gru fusion----
            if self.cfg.FUSION.FUSION_ON:
                voxel_dim = feat.shape[-1]
                up_coords, feat_all, tsdf_target, occ_target = self.gru_fusion(up_coords, feat_all, inputs, i)
                feat = feat_all[:, :voxel_dim]  # voxel feats

                if self.cfg.FUSION.FULL:
                    grid_mask = torch.ones_like(feat[:, 0]).bool()

            tsdf = self.tsdf_preds[i](feat)  # linear  (voxel_num, 1)
            occ = self.occ_preds[i](feat)  # linear  (voxel_num, 1)

            # visualization
            if 'rgb_list' in inputs:
                xyzrgb_targets = self.get_xyzrgb_targets(up_coords, inputs, scale)
                if i < 0:
                    print('stage: ', i)
    
                    for b in range(bs):
                        batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1)
    
                        up_coords_numpy = up_coords[batch_ind][:, 1:].detach().cpu().numpy()
                        occ_target_numpy = occ_target[batch_ind].long().detach().cpu().numpy()
                        tsdf_target_numpy = 1 - tsdf_target[batch_ind].abs().detach().cpu().numpy()
                        # print('tsdf: ', (tsdf_target_numpy.reshape(-1) > 0).sum())
                        # print('occ: ', occ_target_numpy.sum())
    
                        rgb_numpy = xyzrgb_targets[batch_ind].detach().cpu().numpy()
                        visualize_mesh(rgb_numpy, type='rgb')
    
                        visualize_mesh(up_coords_numpy[tsdf_target_numpy.reshape(-1) > 0], tsdf_target_numpy[tsdf_target_numpy.reshape(-1) > 0], type='tsdf')
                        # visualize_mesh(up_coords_numpy[np.bool_(occ_target_numpy.reshape(-1))], tsdf_target_numpy[np.bool_(occ_target_numpy.reshape(-1))], type='tsdf')
                        # visualize_mesh(up_coords_numpy[(tsdf_target_numpy.reshape(-1) > 0) & (occ_target_numpy.reshape(-1) > 0)],
                        #                tsdf_target_numpy[(tsdf_target_numpy.reshape(-1) > 0) & (occ_target_numpy.reshape(-1) > 0)], type='tsdf')
                        visualize_mesh(up_coords_numpy, occ_target_numpy, type='semantic')

            # -------compute loss-------
            if tsdf_target is not None:
                # shape = (24, 24, 24)
                # tsdf2mesh(up_coords[:, 1:] / (2 ** (2 - i)), tsdf, tuple(x * (2 ** i) for x in shape), 'results/tsdf_pred.ply')
                # tsdf2mesh(up_coords[:, 1:] / (2 ** (2 - i)), tsdf_target, tuple(x * (2 ** i) for x in shape), 'results/tsdf_target.ply')

                loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target, mask=grid_mask, pos_weight=self.cfg.POS_WEIGHT)

            else:
                loss = torch.Tensor(np.array([0]))[0]
            loss_dict.update({f'tsdf_occ_loss_{i}': loss})

            # ------define the sparsity for the next stage-----
            occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[i]

            occupancy[grid_mask == False] = False

            # if num == 0:
            #     logger.warning('no valid points: scale {}'.format(i))
            #     return outputs, loss_dict

            # ------avoid out of memory: sample points if num of points is too large-----
            exceed_num = 1.5
            for batch in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == batch).squeeze(1)
                # print(occ_target[batch_ind].sum())
                num_batch = int(occupancy[batch_ind].sum().data.cpu())

                if num_batch < 500:
                    logger.warning('no valid points: scale {}'.format(i))
                    return outputs, loss_dict

                if self.training and num_batch > self.cfg.TRAIN_NUM_SAMPLE[i] * exceed_num:
                    logger.warning('exceed too many points: scale {} num_batch {}'.format(i, num_batch))
                    return outputs, loss_dict

                elif self.training and num_batch > self.cfg.TRAIN_NUM_SAMPLE[i]:
                    logger.warning('choice too many points: scale {} num_batch {}'.format(i, num_batch))
                    choice_batch = np.random.choice(num_batch, num_batch - self.cfg.TRAIN_NUM_SAMPLE[i], replace=False)
                    ind_batch = torch.nonzero(occupancy[batch_ind])
                    temp_exceed = occupancy[batch_ind]
                    temp_exceed[ind_batch[choice_batch]] = False
                    occupancy[batch_ind] = temp_exceed
                    del temp_exceed

            for batch in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == batch).squeeze(1)
                if occ_target[batch_ind][occupancy[batch_ind]].sum() == 0:
                    logger.warning('occ_target is 0: scale {} num_batch {}'.format(i, occ_target[batch_ind][occupancy[batch_ind]].sum()))
                    return outputs, loss_dict

            pre_coords = up_coords[occupancy]
            for b in range(bs):
                batch_ind = torch.nonzero(pre_coords[:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:
                    logger.warning('no valid points: scale {}, batch {}'.format(i, b))
                    return outputs, loss_dict

            pre_feat = feat[occupancy]
            pre_tsdf = tsdf[occupancy]
            pre_occ = occ[occupancy]

            # panoptic
            panoptic_voxel_feats.append(feat_all[occupancy])
            panoptic_coords.append(pre_coords)

            pre_feat = torch.cat([pre_feat, pre_tsdf, pre_occ], dim=1)  # concat feats, tsdf 和 occ

            if i == self.cfg.N_LAYER - 1:
                outputs['coords'] = pre_coords
                outputs['tsdf'] = pre_tsdf

        # TODO ---------------------------------------------------------------------------------------------------------
        # TODO ---------------------------------------   Panoptic Segmentation  ----------------------------------------
        # TODO ---------------------------------------------------------------------------------------------------------
        down_masks_1, down_masks_0 = [], []
        for batch in range(bs):
            batch_ind = []
            for p in range(3):
                batch_ind.append(torch.nonzero(panoptic_coords[p][:, 0] == batch).squeeze(1))

            down_coords = torch.floor_divide(panoptic_coords[2][batch_ind[2]][:, 1:], 2) * 2  # Reduce the coordinates to the number closest to a multiple of 2
            down_coords = torch.cat([panoptic_coords[2][batch_ind[2]][:, 0].unsqueeze(1), down_coords], dim=1)
            down_coords = torch.unique(down_coords, dim=0)
            down_masks = (panoptic_coords[1][batch_ind[1]].unsqueeze(1) == down_coords.unsqueeze(0)).all(dim=2).any(dim=1)  # Find the same points
            down_masks_1.append(down_masks)

            down_coords_b = torch.floor_divide(down_coords[:, 1:], 4) * 4
            down_coords_b = torch.cat([down_coords[:, 0].unsqueeze(1), down_coords_b], dim=1)
            down_coords_b = torch.unique(down_coords_b, dim=0)
            down_masks = (panoptic_coords[0][batch_ind[0]].unsqueeze(1) == down_coords_b.unsqueeze(0)).all(dim=2).any(dim=1)
            down_masks_0.append(down_masks)

        down_masks_1 = torch.cat(down_masks_1, dim=0)
        down_masks_0 = torch.cat(down_masks_0, dim=0)

        # update scale 1
        panoptic_coords[1] = panoptic_coords[1][down_masks_1]
        panoptic_voxel_feats[1] = panoptic_voxel_feats[1][down_masks_1]
        # update scale 0
        panoptic_coords[0] = panoptic_coords[0][down_masks_0]
        panoptic_voxel_feats[0] = panoptic_voxel_feats[0][down_masks_0]

        del down_coords, down_coords_b, down_masks, down_masks_1, down_masks_0

        # voxel_feats transfer
        for p in range(3):
            panoptic_voxel_feats[p] = self.panoptic_preds[p](panoptic_voxel_feats[p])

        panoptic_shape = (96, 96, 96)
        panoptic_outs = []
        for batch in range(bs):
            batch_ind = []
            for p in range(3):
                batch_ind.append(torch.nonzero(panoptic_coords[p][:, 0] == batch).squeeze(1))

            mask_features = self.panoptic_feat_fusion.generate_mask_features(panoptic_feats=panoptic_voxel_feats[2][batch_ind[2]],
                                                                             coords_b=torch.zeros_like(panoptic_coords[2][batch_ind[2]][:, 0]),
                                                                             coords_xyz=panoptic_coords[2][batch_ind[2]][:, 1:],
                                                                             batch_size=1,
                                                                             spitial_shape=panoptic_shape,)

            panoptic_feats_batch, panoptic_coords_batch = [], []
            for p in range(3):
                panoptic_feats_batch.append(panoptic_voxel_feats[p][batch_ind[p]].unsqueeze(0).permute(0, 2, 1))  # [bs=1, c, N_voxels]
                panoptic_coords_batch.append(panoptic_coords[p][batch_ind[p]][..., 1:].unsqueeze(0))  # [bs=1, N_voxels, 3(xyz)]

            panoptic_out_batch = self.panoptic(panoptic_features=panoptic_feats_batch,  # [bs=1, c, N_voxels]
                                               panoptic_coords=panoptic_coords_batch,  # [bs=1, N_voxels, 3(xyz)]
                                               mask_features=mask_features.unsqueeze(0).permute(0, 2, 1),  # [bs=1, c, N_voxels]
                                               spitial_shape=panoptic_shape,)

            panoptic_outs.append(panoptic_out_batch)

        "--- panoptic loss ---"
        panoptic_losses, panoptic_predictions = [], []
        
        # prediction
        if 'rgb_list' in inputs:
            panoptic_targets = self.get_panoptic_targets(panoptic_coords[2], inputs, scale, bs)
        for batch in range(bs):
            batch_ind = torch.nonzero(panoptic_coords[2][:, 0] == batch).squeeze(1)
            panoptic_predictions.append(panoptic_post(panoptic_outs[batch]))
            # # visualize
            # generate_mesh(panoptic_coords[2][batch_ind][:, 1:], panoptic_targets[batch], panoptic_shape, 'results/panoptic_target.ply', mode='panoptic')
            # generate_mesh(panoptic_coords[2][batch_ind][:, 1:], panoptic_predictions[batch]['panoptic_seg'][0], panoptic_shape, 'results/panoptic_pred.ply', mode='panoptic')
        outputs['panoptic_info'] = panoptic_predictions

        # Only calculate supervision within occ_target
        occ_target_occupancy = occ_target[occupancy].view(-1)
        for batch in range(bs):
            batch_ind = torch.nonzero(panoptic_coords[2][:, 0] == batch).squeeze(1)
            panoptic_outs[batch]['pred_masks'] = panoptic_outs[batch]['pred_masks'][..., occ_target_occupancy[batch_ind]]
            for aux_out in panoptic_outs[batch]['aux_outputs']:
                aux_out['pred_masks'] = aux_out['pred_masks'][..., occ_target_occupancy[batch_ind]]

        panoptic_coords[2] = panoptic_coords[2][occ_target_occupancy]
        if 'rgb_list' in inputs:
            panoptic_targets = self.get_panoptic_targets(panoptic_coords[2], inputs, scale, bs)
    
            for batch in range(bs):
                panoptic_losses.append([])
    
                # bipartite matching-based loss
                panoptic_losses[-1] = self.criterion(panoptic_outs[batch], [panoptic_targets[batch]])  # loss per voxel
                for k in list(panoptic_losses[-1].keys()):
                    if k in self.criterion.weight_dict:
                        panoptic_losses[-1][k] *= self.criterion.weight_dict[k]
                    else:
                        panoptic_losses[-1].pop(k)
                panoptic_losses[-1] = sum(panoptic_losses[-1].values()) / 3
    
            if len(panoptic_losses) == 0:
                panoptic_losses = torch.Tensor([0.0]).cuda()[0] * panoptic_outs[batch][
                    'pred_masks'].sum()
            else:
                panoptic_losses = sum(panoptic_losses) / len(panoptic_losses)  # loss per bs per voxel
    
            if panoptic_losses == 0:
                logger.warning('no valid points after panoptic loss')
    
            loss_dict.update({f'panoptic_loss': panoptic_losses})

        return outputs, loss_dict

    @staticmethod
    def compute_loss_init(coord_init, occ_init, tsdf_init_target, occ_init_target):

        occ_init = occ_init.view(-1)
        tsdf_init_target = tsdf_init_target.view(-1)
        occ_init_target = occ_init_target.view(-1)

        valid_target = (tsdf_init_target == 0) | (occ_init_target == 1)
        occ_init = occ_init[valid_target]
        tsdf_init_target = tsdf_init_target[valid_target]

        # # vis
        # coord_init = coord_init[valid_target]
        # visualize_mesh(coord_init.detach().cpu().numpy(), tsdf_init_target.detach().cpu().numpy(), type='tsdf')

        n_p_init = tsdf_init_target.sum()
        if n_p_init == 0:
            logger.warning('target_init: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * occ_init.sum()

        '--- [-1, 1] ---'
        # w_for_1_init = compute_pos_weight(tsdf_init_target)
        #
        # occ_init = occ_init.sigmoid()
        # occ_init = apply_log_transform(occ_init)
        # tsdf_init_target = apply_log_transform(tsdf_init_target)
        #
        # loss = torch.abs(occ_init - tsdf_init_target)
        # loss[tsdf_init_target > 0] = loss[tsdf_init_target > 0] * w_for_1_init
        # loss = loss.mean()

        '--- {0, 1} ---'
        tsdf_init_target = (tsdf_init_target > 0).float()
        w_for_1_init = compute_pos_weight(tsdf_init_target)
        loss = F.binary_cross_entropy_with_logits(occ_init, tsdf_init_target, pos_weight=w_for_1_init)

        return loss

    @staticmethod
    def compute_loss(tsdf, occ, tsdf_target, occ_target, loss_weight=(1, 1),
                     mask=None, pos_weight=1.0):

        # compute occupancy/tsdf loss
        tsdf = tsdf.view(-1)
        occ = occ.view(-1)
        tsdf_target = tsdf_target.view(-1)
        occ_target = occ_target.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            tsdf = tsdf[mask]
            occ = occ[mask]
            tsdf_target = tsdf_target[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * tsdf.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ bce loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # compute tsdf l1 loss
        # tsdf = tsdf.sigmoid()
        tsdf = apply_log_transform(tsdf[occ_target])
        tsdf_target = apply_log_transform(tsdf_target[occ_target])
        tsdf_loss = torch.mean(torch.abs(tsdf - tsdf_target))

        # compute final loss
        loss = loss_weight[0] * occ_loss + loss_weight[1] * tsdf_loss
        return loss

    @staticmethod
    def compute_semantic_loss(sementic, semantic_targets, occ_target, coords, bs):

        device = semantic_targets.device
        occ_target = occ_target.view(-1)
        if occ_target.sum() == 0:
            logger.warning('target: no valid voxel when computing semantic loss - occ_target')
            return torch.Tensor([0.0]).cuda()[0] * sementic.sum()

        sementic = sementic[occ_target]
        semantic_targets = semantic_targets[occ_target]
        # print('semantic targets origin: ', torch.unique(semantic_targets), semantic_targets.shape)

        # Mapping to 20 categories
        valid_classes = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]).to(device)
        indices = torch.where(torch.isin(semantic_targets, valid_classes))[0]
        sementic = sementic[indices]
        semantic_targets = semantic_targets[indices]
        # print('semantic targets 20: ', torch.unique(semantic_targets), semantic_targets.shape)

        if len(semantic_targets) == 0:
            logger.warning('target: no valid voxel when computing semantic loss - 20')
            return torch.Tensor([0.0]).cuda()[0] * sementic.sum()

        mapped_classes = torch.arange(1, 21).to(device)
        mapped_values = torch.searchsorted(valid_classes, semantic_targets)
        semantic_targets = mapped_classes[mapped_values]
        # print('semantic targets 20 mapped: ', torch.unique(semantic_targets), semantic_targets.shape)

        loss = F.cross_entropy(sementic, semantic_targets)

        # coords = coords[occ_target][indices]
        # for b in range(bs):
        #     batch_ind = torch.nonzero(coords[:, 0] == b).squeeze(1)
        #
        #     coords_numpy = coords[batch_ind][:, 1:].detach().cpu().numpy()
        #     semantic_targets_numpy = semantic_targets[batch_ind].long().detach().cpu().numpy()
        #
        #     vis_preds = torch.argmax(sementic[batch_ind].softmax(-1), dim=-1)
        #     semantic_numpy = vis_preds.long().detach().cpu().numpy()
        #     print('semantic pred: ', torch.unique(vis_preds), vis_preds.shape)
        #
        #     visualize_mesh(coords_numpy, semantic_targets_numpy, type='semantic')
        #     visualize_mesh(coords_numpy, semantic_numpy, type='semantic')

        return loss

def tsdf2mesh(coords, tsdf, dim_list, save_path):
    tsdf = tsdf.view(-1).detach()
    tsdf_volume = sparse_to_dense_torch(coords.long(), tsdf, dim_list, 100, tsdf.device, final_tsdf=True)
    tsdf_volume = tsdf_volume.cpu().numpy()

    verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
    mesh.export(save_path)

def sparse_to_dense_torch(locs, values, dim, default_val, device, final_tsdf=False):

    if final_tsdf:
        # default_val = 100
        # dense = torch.full([dim[0], dim[1], dim[2]], default_val, device=device).to(values.dtype)
        # if locs.shape[0] > 0:
        #     dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
        #
        # mask = dense == default_val
        #
        # dense, mask = dense.cpu().numpy(), mask.cpu().numpy()
        # distance, indices = ndimage.distance_transform_edt(mask, return_indices=True)
        # nearest_values = dense[tuple(indices)]
        # dense[mask] = np.where(nearest_values[mask] >= 0, 1, -1)
        # dense = torch.from_numpy(dense).to(device)

        default_val = 1
        dense = torch.full([dim[0], dim[1], dim[2]], default_val, device=device).to(values.dtype)
        if locs.shape[0] > 0:
            dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values

    else:
        dense = torch.full([dim[0], dim[1], dim[2]], float(default_val), device=device).to(values.dtype)
        if locs.shape[0] > 0:
            dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values

    return dense

def compute_pos_weight(targets):
    """
    targets: [N,] bool
    """

    n_all = targets.reshape(-1).shape[0]
    n_p = targets.reshape(-1).sum()

    pos_weight = (n_all - n_p).float() / n_p

    return pos_weight

def occ_overlap(occ_1, occ_2):
    occ_1 = occ_1.view(-1)
    occ_2 = occ_2.view(-1)

    I = torch.sum(occ_1 & occ_2)
    U = torch.sum(occ_1 | occ_2)

    overlap = I / U

    return overlap


def generate_mesh(coords, input, dim_list, save_path=None, mode='tsdf', export_mesh=False):
    if mode == 'tsdf':
        default_val = 1
        input = input.view(-1).detach()
        tsdf_volume = sparse_to_dense_torch(coords.long(), input, dim_list, default_val, input.device, final_tsdf=True)
        tsdf_volume = tsdf_volume.cpu().numpy()

        verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0)
        if export_mesh:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
            mesh.export(save_path)

        return verts

    elif mode == 'panoptic':
        if isinstance(input, dict):
            target_label = input['labels'].int()
            target_instance = torch.arange(1, len(target_label) + 1).int().to(target_label.device)
            target_mask = input['masks'].int()
            first_true_indices = torch.argmax(target_mask, dim=0)
            # panoptic_mask = target_label[first_true_indices]
            panoptic_mask = target_instance[first_true_indices]
            panoptic_mask = panoptic_mask.detach().cpu().numpy()

        else:
            panoptic_mask = input.detach().cpu().numpy()

        visualize_mesh(coords.to(torch.int32).detach().cpu().numpy(), panoptic_mask, type='instance')


