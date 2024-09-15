import torch
import torch.nn as nn
from torchsparse.tensor import PointTensor
from utils import sparse_to_dense_channel, sparse_to_dense_torch
from .modules import ConvGRU


class GRUFusion(nn.Module):

    def __init__(self, cfg, ch_in=None, direct_substitute=False, trianing=True, ch_voxel=None):
        super(GRUFusion, self).__init__()
        self.cfg = cfg
        # replace tsdf in global tsdf volume by direct substitute corresponding voxels
        self.direct_substitude = direct_substitute
        self.trianing = trianing

        if direct_substitute:
            # tsdf
            self.ch_in = [1, 1, 1]
            self.feat_init = 1
        else:
            # features
            self.ch_in = ch_in
            self.feat_init = 0

        if ch_voxel is not None:
            self.ch_voxel = ch_voxel  # voxel feats dim
            self.ch_img = [x - y for x, y in zip(ch_in, ch_voxel)]  # img feats dim

        self.n_scales = len(cfg.THRESHOLDS) - 1
        self.scene_name = [None, None, None]
        self.global_origin = [None, None, None]
        self.global_volume = [None, None, None]
        self.global_img_volume = [None, None, None]
        self.target_tsdf_volume = [None, None, None]

        self.global_instance = [None]  # instance_id
        self.global_semantic = [None]  # semantic_id

        if direct_substitute:
            self.fusion_nets_voxel = None
            self.fusion_nets_img = None
        else:
            self.fusion_nets_voxel = nn.ModuleList()
            self.fusion_nets_img = nn.ModuleList()

            for i, ch in enumerate(self.ch_voxel):
                self.fusion_nets_voxel.append(ConvGRU(hidden_dim=ch,
                                                      input_dim=ch,
                                                      pres=1,
                                                      vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i)))

            for i, ch in enumerate(self.ch_img):
                self.fusion_nets_img.append(ConvGRU(hidden_dim=ch,
                                                    input_dim=ch,
                                                    pres=1,
                                                    vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i)))

    def reset(self, i):
        self.global_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.global_img_volume[i] = torch.Tensor([]).cuda()
        self.target_tsdf_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()

        self.global_instance = torch.empty(0).cuda()  # instance_id
        self.global_semantic = torch.empty(0).cuda()  # semantic_id

    def convert2dense(self, current_coords, current_values, coords_target_global, tsdf_target, relative_origin, scale=2, current_img=None):
        # previous frame
        global_coords = self.global_volume[scale].C
        global_value = self.global_volume[scale].F
        global_tsdf_target = self.target_tsdf_volume[scale].F
        global_coords_target = self.target_tsdf_volume[scale].C

        # dim = (torch.Tensor(self.cfg.N_VOX).cuda() // 2 ** (self.cfg.N_LAYER - scale - 1)).int()
        dim = (torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** (self.cfg.N_LAYER - scale - 1), rounding_mode='floor')).int()
        dim_list = dim.data.cpu().numpy().tolist()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin
        valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)
        if self.cfg.FUSION.FULL is False:
            valid_volume = sparse_to_dense_torch(current_coords, 1, dim_list, 0, global_value.device)
            value = valid_volume[global_coords[valid][:, 0], global_coords[valid][:, 1], global_coords[valid][:, 2]]
            all_true = valid[valid]
            all_true[value == 0] = False
            valid[valid] = all_true
        # sparse to dense
        global_volume = sparse_to_dense_channel(global_coords[valid], global_value[valid], dim_list, self.ch_in[scale], self.feat_init, global_value.device)
        current_volume = sparse_to_dense_channel(current_coords, current_values, dim_list, self.ch_in[scale], self.feat_init, global_value.device)

        if self.cfg.FUSION.FULL is True:
            # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
            if self.direct_substitude:
                updated_coords = torch.nonzero((global_volume.abs() < 1).any(-1) | (current_volume.abs() < 1).any(-1))  # tsdf
            else:
                updated_coords = torch.nonzero((global_volume != 0).any(-1) | (current_volume != 0).any(-1))  # feats
        else:
            updated_coords = current_coords

        # fuse ground truth
        if tsdf_target is not None:
            # mask voxels that are out of the FBV
            global_coords_target = global_coords_target - relative_origin
            valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
            # combine current tsdf and global tsdf
            coords_target = torch.cat([global_coords_target[valid_target], coords_target_global])[:, :3]
            tsdf_target = torch.cat([global_tsdf_target[valid_target], tsdf_target.unsqueeze(-1)])
            # sparse to dense
            target_volume = sparse_to_dense_channel(coords_target, tsdf_target, dim_list, 1, 1,
                                                    tsdf_target.device)
        else:
            target_volume = valid_target = None

        return updated_coords, current_volume, global_volume, target_volume, valid, valid_target

    def compute_overlap(self, points1, points2):
        # points1: [M, 3], points2: [N, 3]
        S = len(points1) + len(points2)
        points1 = points1.unsqueeze(1)
        points2 = points2.unsqueeze(0)

        distances_squared = torch.sum((points1 - points2) ** 2, dim=2)
        same_points_mask = distances_squared == 0

        I = num_same_points = same_points_mask.sum()
        U = S - I
        overlap = I / U  # IOU

        # overlap = I

        return overlap

    def panoptic_fusion(self, scale, global_valid, relative_origin, panoptic_info, current_coords):

        stuff_instance_id_all = [1, 2]  # 1: wall, 2: floor, 0: empty
        max_stuff = max(stuff_instance_id_all)

        overlap_threshold = 0.05

        current_coords = current_coords + relative_origin
        current_voxel_id_all = panoptic_info['panoptic_seg'][0]

        global_instance = self.global_instance[global_valid]  # [N, ] instance_id
        global_semantic = self.global_semantic[global_valid]  # [N, ] semantic_id
        if len(self.global_instance) > 0:
            max_global_instance_id = torch.max(self.global_instance)
            if max_global_instance_id <= max_stuff:
                max_global_instance_id = max_stuff
        else:
            max_global_instance_id = max_stuff

        new_current_instance = torch.zeros_like(current_voxel_id_all)  # [N, ]
        new_current_semantic = torch.zeros_like(current_voxel_id_all)  # [N, ]

        num_current_instance = len(panoptic_info['panoptic_seg'][1])
        increment_count = 1
        for i in range(num_current_instance):
            cls = panoptic_info['panoptic_seg'][1][i]['category_id']

            if panoptic_info['panoptic_seg'][1][i]['isthing']:
                if cls in global_semantic:
                    cls_global_index = global_semantic == cls  # in current volume
                    ins_global = global_instance[cls_global_index]
                    ins_global_all = torch.unique(ins_global)

                    match_flag = False

                    for ins_id in ins_global_all:
                        comparison_global_index = self.global_instance == ins_id  # in global volume
                        comparison_global_coords = self.global_volume[scale].C[comparison_global_index.view(-1)]

                        overlap = self.compute_overlap(comparison_global_coords, current_coords[current_voxel_id_all == i+1])
                        if overlap > overlap_threshold:
                            new_current_instance[current_voxel_id_all == i+1] = int(ins_id)
                            new_current_semantic[current_voxel_id_all == i+1] = cls
                            match_flag = True
                            break
                    if match_flag == False:
                        new_current_instance[current_voxel_id_all == i+1] = int(max_global_instance_id + increment_count)
                        new_current_semantic[current_voxel_id_all == i+1] = cls
                        increment_count = increment_count + 1

                else:
                    new_current_instance[current_voxel_id_all == i+1] = int(max_global_instance_id + increment_count)
                    new_current_semantic[current_voxel_id_all == i+1] = cls
                    increment_count = increment_count + 1

            # stuff
            else:
                new_current_instance[current_voxel_id_all == i+1] = cls
                new_current_semantic[current_voxel_id_all == i+1] = cls

        return new_current_instance, new_current_semantic

    def update_map(self, value, coords, target_volume, valid, valid_target, relative_origin, scale,
                   new_current_instance=None, new_current_semantic=None):
        # pred
        self.global_volume[scale].F = torch.cat([self.global_volume[scale].F[valid == False], value])
        coords = coords + relative_origin
        self.global_volume[scale].C = torch.cat([self.global_volume[scale].C[valid == False], coords])

        if self.direct_substitude:
            self.global_instance = torch.cat([self.global_instance[valid == False], new_current_instance])
            self.global_semantic = torch.cat([self.global_semantic[valid == False], new_current_semantic])

        # target
        if target_volume is not None:
            target_volume = target_volume.squeeze()
            self.target_tsdf_volume[scale].F = torch.cat(
                [self.target_tsdf_volume[scale].F[valid_target == False],
                 target_volume[target_volume.abs() < 1].unsqueeze(-1)])
            target_coords = torch.nonzero(target_volume.abs() < 1) + relative_origin

            self.target_tsdf_volume[scale].C = torch.cat(
                [self.target_tsdf_volume[scale].C[valid_target == False], target_coords])

    def save_mesh(self, scale, outputs, scene):
        if outputs is None:
            outputs = dict()
        if "scene_name" not in outputs:
            outputs['origin'] = []
            outputs['scene_tsdf'] = []
            outputs['scene_name'] = []
            outputs['scene_instance'] = []
            outputs['scene_semantic'] = []

        # only keep the newest result
        if scene in outputs['scene_name']:
            # delete old
            idx = outputs['scene_name'].index(scene)
            del outputs['origin'][idx]
            del outputs['scene_tsdf'][idx]
            del outputs['scene_name'][idx]
            del outputs['scene_instance'][idx]
            del outputs['scene_semantic'][idx]

        # scene name
        outputs['scene_name'].append(scene)

        fuse_coords = self.global_volume[scale].C
        tsdf = self.global_volume[scale].F.squeeze(-1)
        instance = self.global_instance.squeeze(-1)
        semantic = self.global_semantic.squeeze(-1)
        max_c = torch.max(fuse_coords, dim=0)[0][:3]
        min_c = torch.min(fuse_coords, dim=0)[0][:3]
        outputs['origin'].append(min_c * self.cfg.VOXEL_SIZE * (2 ** (self.cfg.N_LAYER - scale - 1)))

        ind_coords = fuse_coords - min_c
        dim_list = (max_c - min_c + 1).int().data.cpu().numpy().tolist()
        tsdf_volume = sparse_to_dense_torch(ind_coords, tsdf, dim_list, 1, tsdf.device)
        instance_volume = sparse_to_dense_torch(ind_coords, instance, dim_list, 0, instance.device)
        semantic_volume = sparse_to_dense_torch(ind_coords, semantic, dim_list, 0, semantic.device)
        outputs['scene_tsdf'].append(tsdf_volume)
        outputs['scene_instance'].append(instance_volume)
        outputs['scene_semantic'].append(semantic_volume)

        return outputs

    def forward(self, coords, values_in, inputs, scale=2, outputs=None, save_mesh=False, panoptic_infos=None):
        if self.global_volume[scale] is not None:
            # delete computational graph to save memory
            self.global_volume[scale] = self.global_volume[scale].detach()
            self.global_img_volume[scale] = self.global_img_volume[scale].detach()

        batch_size = len(inputs['fragment'])
        interval = 2 ** (self.cfg.N_LAYER - scale - 1)

        tsdf_target_all = None
        occ_target_all = None
        panoptic_img_feats_all = None
        values_all = None
        updated_coords_all = None

        # ---incremental fusion----
        for i in range(batch_size):
            scene = inputs['scene'][i]  # scene name
            global_origin = inputs['vol_origin'][i]
            origin = inputs['vol_origin_partial'][i]

            if scene != self.scene_name[scale] and self.scene_name[scale] is not None and self.direct_substitude:
                outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

            if self.scene_name[scale] is None or scene != self.scene_name[scale]:
                self.scene_name[scale] = scene
                self.reset(scale)
                self.global_origin[scale] = global_origin

            # each level has its corresponding voxel size
            voxel_size = self.cfg.VOXEL_SIZE * interval

            # relative origin in global volume
            relative_origin = (origin - self.global_origin[scale]) / voxel_size
            relative_origin = relative_origin.cuda().long()

            batch_ind = torch.nonzero(coords[:, 0] == i).squeeze(1)
            if len(batch_ind) == 0:
                continue
            # coords_b = coords[batch_ind, 1:].long() // interval
            coords_b = torch.div(coords[batch_ind, 1:].long(), interval, rounding_mode='floor')
            values = values_in[batch_ind]

            if 'occ_list' in inputs.keys():
                # get partial gt
                occ_target = inputs['occ_list'][self.cfg.N_LAYER - scale - 1][i]
                tsdf_target = inputs['tsdf_list'][self.cfg.N_LAYER - scale - 1][i][occ_target]
                coords_target = torch.nonzero(occ_target)
            else:
                coords_target = tsdf_target = None

            # convert to dense: 1. convert sparse feature to dense feature; 2. combine current feature coordinates and
            # previous feature coordinates within FBV from our backend map to get new feature coordinates (updated_coords)
            updated_coords, current_volume, global_volume, target_volume, valid, valid_target = self.convert2dense(
                coords_b,
                values,  # values就是feats
                coords_target,
                tsdf_target,
                relative_origin,
                scale)

            # dense to sparse: get features using new feature coordinates (updated_coords)  sparse: (x_dim*y_dim*z_dim, ), dense: (x_dim, y_dim, z_dim, c)
            values = current_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]  # (voxel_num, c)
            global_values = global_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]  # (voxel_num, c)
            # get fused gt
            if target_volume is not None:
                tsdf_target = target_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]  # (voxel_num, 1)
                occ_target = tsdf_target.abs() < 1  # (voxel_num, 1)
            else:
                tsdf_target = occ_target = None

            if not self.direct_substitude:  # GRUFusion
                # convert to aligned camera coordinate
                r_coords = updated_coords.detach().clone().float()
                r_coords = r_coords.permute(1, 0).contiguous().float() * voxel_size + origin.unsqueeze(-1).float()
                r_coords = torch.cat((r_coords, torch.ones_like(r_coords[:1])), dim=0)
                r_coords = inputs['world_to_aligned_camera'][i, :3, :] @ r_coords
                r_coords = torch.cat([r_coords, torch.zeros(1, r_coords.shape[-1]).to(r_coords.device)])
                r_coords = r_coords.permute(1, 0).contiguous()

                # TODO ConvGRU
                # voxel feats
                h_voxel = PointTensor(global_values[:, :self.ch_voxel[scale]], r_coords)  # global_values:[N_voxels, c], r_coords:[N_voxels, 4]
                x_voxel = PointTensor(values[:, :self.ch_voxel[scale]], r_coords)  # values:[N_voxels, c], r_coords:[N_voxels, 4]
                values_voxel = self.fusion_nets_voxel[scale](h_voxel, x_voxel)

                h_img = PointTensor(global_values[:, self.ch_voxel[scale]:], r_coords)  # global_values:[N_voxels, c], r_coords:[N_voxels, 4]
                x_img = PointTensor(values[:, self.ch_voxel[scale]:], r_coords)  # values:[N_voxels, c], r_coords:[N_voxels, 4]
                values_img = self.fusion_nets_img[scale](h_img, x_img)

                values = torch.cat([values_voxel, values_img], dim=-1)
                del h_voxel, x_voxel, values_voxel, h_img, x_img, values_img

            if self.direct_substitude:
                dim = (torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** (self.cfg.N_LAYER - scale - 1), rounding_mode='floor')).int()
                dim_list = dim.data.cpu().numpy().tolist()
                current_instance_volume = sparse_to_dense_channel(coords_b, panoptic_infos[i]['panoptic_seg'][0].unsqueeze(1), dim_list, 1, 0, values.device)
                panoptic_infos[i]['panoptic_seg'][0] = current_instance_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]].squeeze(1)

                # new_current_instance: [N, ], new_current_semantic: [N, ]
                new_current_instance, new_current_semantic = self.panoptic_fusion(scale=scale,
                                                                                  global_valid=valid,
                                                                                  relative_origin=relative_origin,
                                                                                  panoptic_info=panoptic_infos[i],
                                                                                  current_coords=updated_coords)

            # feed back to global volume (direct substitute)
            if not self.direct_substitude:
                self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, scale)
            else:
                self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, scale,
                                new_current_instance.view(-1, 1), new_current_semantic.view(-1, 1))

            if updated_coords_all is None:
                updated_coords_all = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                               dim=1)
                values_all = values
                tsdf_target_all = tsdf_target
                occ_target_all = occ_target

            else:
                updated_coords = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                           dim=1)
                updated_coords_all = torch.cat([updated_coords_all, updated_coords])
                values_all = torch.cat([values_all, values])
                if tsdf_target_all is not None:
                    tsdf_target_all = torch.cat([tsdf_target_all, tsdf_target])
                    occ_target_all = torch.cat([occ_target_all, occ_target])

            if self.direct_substitude and save_mesh:
                outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

        if self.direct_substitude:
            return outputs
        else:
            return updated_coords_all, values_all, tsdf_target_all, occ_target_all
