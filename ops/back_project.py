import torch
from torch.nn.functional import grid_sample


def back_project(coords, origin, voxel_size, feats, KRcam, min_view_number):

    n_views, bs, c, h, w = feats.shape

    coords_all = torch.empty(0, 4).cuda()
    feature_volume_all = torch.empty(0, c + 1).cuda()
    count = torch.zeros(coords.shape[0]).cuda()

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

        # mask > min_view_number
        count[batch_ind] = mask.sum(dim=0).float()
        valid_voxel = count[batch_ind] >= min_view_number  # min:2, max:9
        num_valid = torch.sum(valid_voxel)
        if num_valid == 0:
            return None

        im_grid = im_grid[:, valid_voxel, :]
        mask = mask[:, valid_voxel]
        im_z = im_z[:, valid_voxel]

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)

        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)

        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        # aggregate multi view
        features = features.sum(dim=0)
        mask = mask.sum(dim=0)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1
        in_scope_mask = mask.unsqueeze(0)
        features /= in_scope_mask
        features = features.permute(1, 0).contiguous()  # [N_voxels, dim]

        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1)  # feats

        feature_volume_all = torch.concat([feature_volume_all, features], dim=0)
        coords_all = torch.concat([coords_all, coords[batch_ind][valid_voxel]], dim=0)

    return [feature_volume_all, coords_all, count]









