import time
import spconv.pytorch as spconv
import torch
from torch.nn.functional import grid_sample
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules import (SparseSubMConv3d, SparseConv3d, Linear4xTrans, Fusion_Block, Conv2d_Residual_Block,
                            Conv2d_Block, Spares3dELAN, SparseConv3d_Residual)

class Occupancy_Initialization(nn.Module):
    def __init__(self, ch_initialization_all, ch_initialization_down, n_views):
        super(Occupancy_Initialization, self).__init__()

        ch_all = ch_initialization_all[0] + ch_initialization_all[1] + ch_initialization_all[2]
        self.self_fusion_1x = Fusion_Block(ch_initialization_all[0])
        self.self_fusion_2x = Fusion_Block(ch_initialization_all[1])
        self.self_fusion_4x = Fusion_Block(ch_initialization_all[2])
        # self.pool2x = nn.AvgPool2d(2)
        self.pool4x = nn.AvgPool2d(2)

        self.fusion_down = Conv2d_Block(ch_all, ch_initialization_down, 1)
        self.post_fusion_1 = Conv2d_Residual_Block(ch_initialization_down, 3)
        self.post_fusion_2 = Conv2d_Residual_Block(ch_initialization_down, 3)
        self.post_fusion_3 = Conv2d_Residual_Block(ch_initialization_down, 3)
        self.post_fusion_4 = Conv2d_Residual_Block(ch_initialization_down, 3)

        self.similary_1 = Spares3dELAN(ch_initialization_down)
        self.norm0 = nn.BatchNorm1d(ch_initialization_down)
        self.subm1 = SparseSubMConv3d(ch_initialization_down, ch_initialization_down, 3)
        self.norm1 = nn.LayerNorm(ch_initialization_down)
        self.subm2 = SparseSubMConv3d(ch_initialization_down, ch_initialization_down, 3)
        self.norm2 = nn.LayerNorm(ch_initialization_down)
        self.subm3 = SparseSubMConv3d(ch_initialization_down, ch_initialization_down, 3)
        self.norm3 = nn.LayerNorm(ch_initialization_down)
        self.subm4 = SparseSubMConv3d(ch_initialization_down, 1, 3)
        self.norm4 = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()


    def feat_fusion_pre(self, feats_1x, feats_2x, feats_4x):
        feats_1x = self.self_fusion_1x(feats_1x)
        feats_2x = self.self_fusion_2x(feats_2x)
        feats_4x = self.self_fusion_4x(feats_4x)

        feats_1x = F.interpolate(feats_1x, scale_factor=2, mode='bilinear')
        # feats_2x = F.interpolate(feats_2x, scale_factor=2, mode='bilinear')
        # feats_2x = self.pool2x(feats_2x)
        feats_4x = self.pool4x(feats_4x)
        feats_fusion = torch.concat([feats_1x, feats_2x, feats_4x], dim=1)
        feats_fusion = self.fusion_down(feats_fusion)  # Extract key features and reduce the amount of calculation

        feats_fusion = self.post_fusion_1(feats_fusion)
        feats_fusion = self.post_fusion_2(feats_fusion)
        feats_fusion = self.post_fusion_3(feats_fusion)
        feats_fusion = self.post_fusion_4(feats_fusion)

        return feats_fusion


    def forward(self, coords, origin, voxel_size, features_all, KRcam, shape, stage, min_view_number):

        feats_1x = torch.stack([feat[2] for feat in features_all])
        feats_2x = torch.stack([feat[1] for feat in features_all])
        feats_4x = torch.stack([feat[0] for feat in features_all])

        if stage == 0:
            n_views, bs, c, h, w = feats_1x.shape
        elif stage == 1:
            n_views, bs, c, h, w = feats_2x.shape
        else:
            n_views, bs, c, h, w = feats_4x.shape

        count = torch.zeros(coords.shape[0]).cuda()

        occ_init = torch.empty(0, 1).to(feats_1x.dtype).cuda()  # [N_voxels, 1]
        coord_init = torch.empty(0, 4).to(coords.dtype).cuda()  # [N_voxels, 4(bxyz)]

        for batch in range(bs):
            batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
            coords_batch = coords[batch_ind][:, 1:]  # coords: [num_voxel, bxyz]

            coords_batch = coords_batch.view(-1, 3)
            origin_batch = origin[batch].unsqueeze(0)
            proj_batch = KRcam[:, batch]

            grid_batch = coords_batch * voxel_size + origin_batch.float()
            rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
            rs_grid = rs_grid.permute(0, 2, 1).contiguous()
            nV = rs_grid.shape[-1]
            rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)

            # Project grid
            im_p = proj_batch @ rs_grid
            im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
            im_x = im_x / im_z
            im_y = im_y / im_z

            # AAfter voxel projection is outside the image field of view, the mask behind the camera is lost. (True: valid, False: invalid)
            im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
            mask = im_grid.abs() <= 1
            mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

            count[batch_ind] = mask.sum(dim=0).float()  # Number of visible views for each voxel
            valid_voxel = count[batch_ind] >= min_view_number  # Filter to have at least two views visible  min:2, max:9
            num_valid = torch.sum(valid_voxel)
            if num_valid < 10 * 10 * 10:  # Some fragments have too few points, and the learning effect is poor
                return None

            im_grid = im_grid[:, valid_voxel, :]
            im_grid = im_grid.view(n_views, 1, -1, 2)
            mask = mask[:, valid_voxel].view(n_views, -1)

            'Similarity calculation adapter while aggregating window features'
            feats_fusion_init = self.feat_fusion_pre(feats_1x[:, batch], feats_2x[:, batch], feats_4x[:, batch])

            dim = feats_fusion_init.shape[1]

            features = grid_sample(feats_fusion_init, im_grid, padding_mode='zeros', align_corners=True)

            features = features.view(n_views, dim, -1)  # [N_views, dim, N_voxels]
            features[mask.unsqueeze(1).expand(-1, dim, -1) == False] = 0
            features = features.permute(2, 0, 1)  # [N_voxels, N_views, dim]

            "Variance"
            mask = mask.transpose(0, 1)
            mean = (mask.unsqueeze(2) * features).sum(1) / mask.sum(1).unsqueeze(1)
            var = (mask.unsqueeze(2) * ((features - mean.unsqueeze(1)) ** 2)).sum(1) / mask.sum(1).unsqueeze(1)

            # sparse 3d conv
            subm_coords = coords[batch_ind][valid_voxel] / (2 ** (2 - stage))
            subm_coords = subm_coords.to(coords.dtype)
            subm_coords[:, 0] = 0

            voxel_3d = var
            voxel_3d = self.norm0(voxel_3d)
            voxel_3d = self.similary_1(  # ELAN
                voxel_features_o=voxel_3d,
                voxel_coords_bxyz=subm_coords,
                batch_size=1,
                spitial_shape=shape,
            )

            voxel_3d_residual = self.subm1(features=voxel_3d,
                                           coords=subm_coords,
                                           spitial_shape=shape,
                                           bs=1)
            voxel_3d_residual = self.relu(voxel_3d_residual)
            voxel_3d_residual = voxel_3d_residual + voxel_3d
            voxel_3d_residual = self.norm1(voxel_3d_residual)

            voxel_3d = voxel_3d_residual
            voxel_3d_residual = self.subm2(features=voxel_3d,
                                           coords=subm_coords,
                                           spitial_shape=shape,
                                           bs=1)
            voxel_3d_residual = self.relu(voxel_3d_residual)
            voxel_3d_residual = voxel_3d_residual + voxel_3d
            voxel_3d_residual = self.norm2(voxel_3d_residual)

            voxel_3d = voxel_3d_residual
            voxel_3d_residual = self.subm3(features=voxel_3d,
                                           coords=subm_coords,
                                           spitial_shape=shape,
                                           bs=1)
            voxel_3d_residual = self.relu(voxel_3d_residual)
            voxel_3d_residual = voxel_3d_residual + voxel_3d
            voxel_3d_residual = self.norm3(voxel_3d_residual)

            voxel_3d_residual = self.subm4(features=voxel_3d_residual,
                                           coords=subm_coords,
                                           spitial_shape=shape,
                                           bs=1)
            voxel_3d_residual = self.norm4(voxel_3d_residual)

            occ_init = torch.concat([occ_init, voxel_3d_residual], dim=0)

            coord_init = torch.concat([coord_init, coords[batch_ind][valid_voxel]], dim=0)

        del mean, var, voxel_3d, voxel_3d_residual

        return [occ_init, coord_init, count]


class Back_Project(nn.Module):
    def __init__(self, dim):
        super(Back_Project, self).__init__()

    def forward(self, coords, origin, voxel_size, feats, KRcam, min_view_number):
        '''
        Unproject the image fetures to form a 3D (sparse) feature volume
        '''

        n_views, bs, c, h, w = feats.shape
        num_voxels = coords.shape[0]

        count = torch.zeros(coords.shape[0]).cuda()

        coord_init = torch.empty(0, 4).to(coords.dtype).cuda()  # [N_voxels, 4(bxyz)]
        occ_feat_init = torch.empty(0, c).to(feats.dtype).cuda()  # [N_voxels, c]

        im_grid_all = []
        mask_all = []

        for batch in range(bs):
            batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
            coords_batch = coords[batch_ind][:, 1:]  # coords: [num_voxel, bxyz]

            coords_batch = coords_batch.view(-1, 3)
            origin_batch = origin[batch].unsqueeze(0)
            feats_batch = feats[:, batch]
            proj_batch = KRcam[:, batch]

            grid_batch = coords_batch * voxel_size + origin_batch.float()
            rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
            rs_grid = rs_grid.permute(0, 2, 1).contiguous()
            nV = rs_grid.shape[-1]
            rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)

            # Project grid
            im_p = proj_batch @ rs_grid
            im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
            im_x = im_x / im_z
            im_y = im_y / im_z

            im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
            mask = im_grid.abs() <= 1
            mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

            count[batch_ind] = mask.sum(dim=0).float()
            valid_voxel = count[batch_ind] >= min_view_number
            num_valid = torch.sum(valid_voxel)
            if num_valid == 0:
                return None

            im_grid = im_grid[:, valid_voxel, :]
            im_grid_all.append(im_grid)

            feats_batch = feats_batch.view(n_views, c, h, w)
            im_grid = im_grid.view(n_views, 1, -1, 2)

            features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)

            mask = mask[:, valid_voxel].view(n_views, -1)
            mask_all.append(mask)

            features = features.view(n_views, c, -1)  # [N_views, c, N_voxels]
            features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
            # aggregate multi view
            features = features.sum(dim=0)
            mask = mask.sum(dim=0)
            invalid_mask = mask == 0
            mask[invalid_mask] = 1
            in_scope_mask = mask.unsqueeze(0)
            features /= in_scope_mask
            features = features.permute(1, 0).contiguous()  # [N_voxels, dim]

            occ_feat_init = torch.concat([occ_feat_init, features], dim=0)
            coord_init = torch.concat([coord_init, coords[batch_ind][valid_voxel]], dim=0)

        return [occ_feat_init, coord_init, torch.cat(im_grid_all, dim=1), torch.cat(mask_all, dim=1), count]


def get_img_feats(coords, origin, voxel_size, feats, KRcam, min_view_number):
    n_views, bs, c, h, w = feats.shape
    num_voxels = coords.shape[0]
    count = torch.zeros(coords.shape[0]).cuda()

    img_feats = torch.empty(0, c).to(feats.dtype).cuda()  # [N_voxels, c]

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]  # coords: [num_voxel, bxyz]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        grid_batch = coords_batch * voxel_size + origin_batch.float()
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)

        # Project grid
        im_p = proj_batch @ rs_grid
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        count[batch_ind] = mask.sum(dim=0).float()
        valid_voxel = count[batch_ind] >= min_view_number
        num_valid = torch.sum(valid_voxel)

        im_grid = im_grid[:, valid_voxel, :]

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)

        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)

        mask = mask[:, valid_voxel].view(n_views, -1)

        features = features.view(n_views, c, -1)  # [N_views, c, N_voxels]
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        # aggregate multi view
        features = features.sum(dim=0)
        mask = mask.sum(dim=0)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1
        in_scope_mask = mask.unsqueeze(0)
        features /= in_scope_mask
        features = features.permute(1, 0).contiguous()  # [N_voxels, dim]

        img_feats = torch.concat([img_feats, features], dim=0)

    return img_feats





