import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor
from torchsparse.utils import *
import spconv.pytorch as spconv
from torch.nn import functional as F

from ops.torchsparse_utils import *

__all__ = ['SPVCNN', 'SConv3d', 'ConvGRU']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # x: SparseTensor z: PointTensor
        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        z1 = voxel_to_point(x2, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y3 = point_to_voxel(x2, z1)
        if self.dropout:
            y3.F = self.dropout(y3.F)
        y3 = self.up1[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up1[1](y3)

        y4 = self.up2[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up2[1](y4)
        z3 = voxel_to_point(y4, z1)
        z3.F = z3.F + self.point_transforms[1](z1.F)

        return z3.F


class SConv3d(nn.Module):
    def __init__(self, inc, outc, pres, vres, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = spnn.Conv3d(inc,
                               outc,
                               kernel_size=ks,
                               dilation=dilation,
                               stride=stride)
        self.point_transforms = nn.Sequential(
            nn.Linear(inc, outc),
        )
        self.pres = pres
        self.vres = vres

    def forward(self, z):
        x = initial_voxelize(z, self.pres, self.vres)
        x = self.net(x)
        out = voxel_to_point(x, z, nearest=False)
        out.F = out.F + self.point_transforms(z.F)
        return out


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, pres=1, vres=1):
        super(ConvGRU, self).__init__()
        self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)

    def forward(self, h, x):
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        '''
        hx = PointTensor(torch.cat([h.F, x.F], dim=1), h.C)

        z = torch.sigmoid(self.convz(hx).F)
        r = torch.sigmoid(self.convr(hx).F)
        x.F = torch.cat([r * h.F, x.F], dim=1)
        q = torch.tanh(self.convq(x).F)

        h.F = (1 - z) * h.F + z * q
        return h.F

class SparseConv3d(nn.Module):
    def __init__(self, C_in, C_out, Kernel, Stride, Padding):
        super(SparseConv3d, self).__init__()
        self.sparseconv3d = spconv.SparseConv3d(C_in, C_out, Kernel, Stride, Padding)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.sparseconv3d.weight)
        torch.nn.init.constant_(self.sparseconv3d.bias, 0)

    def forward(self, features, coords, spitial_shape, bs):
        '''
        :param features: voxel features [N, c]
        :param coords: coords [N, 4(bxyz)]
        :return: new_features: [M, c] (M > N)
        '''

        features = spconv.SparseConvTensor(features, coords.to(torch.int32), spitial_shape, bs)  # indices: bxyz
        features = self.sparseconv3d(features)
        new_features = features.features
        new_coords = features.indices

        return new_features, new_coords

class SparseSubMConv3d(nn.Module):
    def __init__(self, C_in, C_out, Kernel, Stride=1):
        super(SparseSubMConv3d, self).__init__()
        self.sparsesubmconv3d = spconv.SubMConv3d(C_in, C_out, Kernel, Stride)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.sparsesubmconv3d.weight)
        torch.nn.init.constant_(self.sparsesubmconv3d.bias, 0)

    def forward(self, features, coords, spitial_shape, bs):
        '''
        :param features: voxel features [N, c]
        :param coords: coords [N, 4(bxyz)]
        :return: new_features: [N, c]
        '''

        features = spconv.SparseConvTensor(features, coords.to(torch.int32), spitial_shape, bs)  # indices: bxyz
        features = self.sparsesubmconv3d(features)
        new_features = features.features

        return new_features

class Linear4xTrans(nn.Module):
    def __init__(self, C_in, C_out):
        super(Linear4xTrans, self).__init__()

        self.linear1 = nn.Linear(C_in, C_in * 4)
        self.norm1 = nn.LayerNorm(C_in * 4)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(C_in * 4, C_in)
        self.norm2 = nn.LayerNorm(C_in)

        self.linear3 = nn.Linear(C_in, C_out)

        self.use_residual = C_in == C_out

        self.weight_initialization()

    def weight_initialization(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out2 = self.linear3(out)
        if self.use_residual:
            out2 = out2 + out

        return out2

class Fusion_Block(nn.Module):
    def __init__(self, C):
        super(Fusion_Block, self).__init__()

        self.conv1 = nn.Conv2d(C, C, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(C)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(C, C, 1, padding='same')
        self.bn2 = nn.BatchNorm2d(C)

        # ELAN
        self.ELAN = ELAN(C)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.ELAN(out)

        return out

class ELAN(nn.Module):
    def __init__(self, dim):
        super(ELAN, self).__init__()

        self.conv1 = Conv2d_Block(dim, dim, 1)

        self.conv2 = Conv2d_Block(dim, dim, 1)
        self.conv3 = Conv2d_Block(dim, int(dim / 2), 3)
        self.conv4 = Conv2d_Block(int(dim / 2), int(dim / 2), 3)
        self.conv5 = Conv2d_Block(int(dim / 2), int(dim / 2), 3)
        self.conv6 = Conv2d_Block(int(dim / 2), int(dim / 2), 3)

        self.conv7 = Conv2d_Block(dim * 4, dim, 1)

    def forward(self, x):
        f = self.conv1(x)
        f2 = self.conv2(x)
        f = torch.concat([f, f2], dim=1)

        f2 = self.conv3(f2)
        f = torch.concat([f, f2], dim=1)
        f2 = self.conv4(f2)
        f = torch.concat([f, f2], dim=1)
        f2 = self.conv5(f2)
        f = torch.concat([f, f2], dim=1)
        f2 = self.conv6(f2)
        f = torch.concat([f, f2], dim=1)

        f = self.conv7(f)

        return f

class Conv2d_Block(nn.Module):
    def __init__(self, C_in, C_out, Kernel):
        super(Conv2d_Block, self).__init__()

        self.conv = nn.Conv2d(C_in, C_out, Kernel, padding='same')
        self.bn = nn.BatchNorm2d(C_out)
        self.act = nn.ReLU()

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))


class Conv2d_Residual_Block(nn.Module):
    def __init__(self, C, Kernel):
        super(Conv2d_Residual_Block, self).__init__()

        self.conv = nn.Conv2d(C, C, Kernel, padding='same')
        self.bn = nn.BatchNorm2d(C)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = out + x
        out = self.bn(out)

        return out

class Spares3dELAN(nn.Module):
    def __init__(self, dim):
        super(Spares3dELAN, self).__init__()

        self.conv1 = SubMconv3dBlock(dim, dim, 1, 1, 0)

        self.conv2 = SubMconv3dBlock(dim, dim, 1, 1, 0)
        self.conv3 = SubMconv3dBlock(dim, int(dim / 2), 3, 1, 1)
        self.conv4 = SubMconv3dBlock(int(dim / 2), int(dim / 2), 3, 1, 1)
        self.conv5 = SubMconv3dBlock(int(dim / 2), int(dim / 2), 3, 1, 1)
        self.conv6 = SubMconv3dBlock(int(dim / 2), int(dim / 2), 3, 1, 1)

        self.conv7 = SubMconv3dBlock(dim * 4, dim, 1, 1, 0)

    def forward(self, voxel_features_o, voxel_coords_bxyz, batch_size, spitial_shape):
        """
        :param voxel_features: [bs*N_voxels, dim]
        :param voxel_coords_bxyz: [bs*N_voxels, 4(bxyz)]
        """
        voxel_features = spconv.SparseConvTensor(voxel_features_o, voxel_coords_bxyz, spitial_shape, batch_size=batch_size)  # indices: bxyz

        f1 = self.conv1(voxel_features)
        f2 = self.conv2(voxel_features)
        voxel_features = voxel_features.replace_feature(f1.features)
        voxel_features = voxel_features.replace_feature(torch.cat([voxel_features.features, f2.features], dim=-1))

        f2 = self.conv3(f2)
        voxel_features = voxel_features.replace_feature(torch.cat([voxel_features.features, f2.features], dim=-1))
        f2 = self.conv4(f2)
        voxel_features = voxel_features.replace_feature(torch.cat([voxel_features.features, f2.features], dim=-1))
        f2 = self.conv5(f2)
        voxel_features = voxel_features.replace_feature(torch.cat([voxel_features.features, f2.features], dim=-1))
        f2 = self.conv6(f2)
        voxel_features = voxel_features.replace_feature(torch.cat([voxel_features.features, f2.features], dim=-1))

        voxel_features = self.conv7(voxel_features)

        return voxel_features.features

class SubMconv3dBlock(nn.Module):
    def __init__(self, C_in, C_out, Kernel, Stride, Padding):
        super(SubMconv3dBlock, self).__init__()

        self.conv = spconv.SubMConv3d(C_in, C_out, Kernel, Stride, Padding)
        self.ln = nn.LayerNorm(C_out)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x.replace_feature(self.act(self.ln(x.features)))

        return x

class Linear_Residual(nn.Module):
    def __init__(self, dim):
        super(Linear_Residual, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.linear(x)  # Linear
        out = self.activation(out)
        out = x + out
        out = self.norm(out)  # Norm

        return out

class SparseConv3d_Residual(nn.Module):
    def __init__(self, dim, Kernel):
        super(SparseConv3d_Residual, self).__init__()
        self.SConv3d = SparseSubMConv3d(dim, dim, Kernel)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, coords, spitial_shape, bs):
        out = self.SConv3d(features=x, coords=coords, spitial_shape=spitial_shape, bs=bs)
        out = self.activation(out)
        out = x + out
        out = self.norm(out)

        return out

# TODO *********************************************  Panoptic_Feat_Fusion   *******************************************
class Panoptic_Feat_Fusion(nn.Module):
    def __init__(self, self_channel, panoptic_channel, ch_initialization):
        super(Panoptic_Feat_Fusion, self).__init__()

        'Basic features of each point (including geometric features and semantic features) to panoramic semantic features of each point'
        # img_feats
        # self.self_fusion_1x = Fusion_Block(ch_initialization[0])
        # self.self_fusion_2x = Fusion_Block(ch_initialization[1])
        # self.self_fusion_4x = Fusion_Block(ch_initialization[2])
        #
        # ch_all = ch_initialization[0] + ch_initialization[1] + ch_initialization[2]
        # self.fusion_down = Conv2d_Block(ch_all, self_channel, 1)
        # self.post_fusion_1 = Conv2d_Residual_Block(self_channel, 3)
        # self.post_fusion_2 = Conv2d_Residual_Block(self_channel, 3)

        self.img2panoptic_0 = nn.Linear(ch_initialization[2], panoptic_channel)
        # self.img2panoptic_1 = Linear_Residual(panoptic_channel)

        # occ_feats
        self.occ2panoptic_0 = nn.Linear(self_channel, panoptic_channel)
        # self.occ2panoptic_1 = Linear_Residual(panoptic_channel)
        # self.occ2panoptic_conv0 = SparseConv3d_Residual(panoptic_channel, 3)
        # self.occ2panoptic_conv1 = SparseConv3d_Residual(panoptic_channel, 3)
        # self.occ2panoptic_conv2 = SparseConv3d_Residual(panoptic_channel, 3)

        # img & occ fusion
        # self.pre_fusion = Linear4xTrans(panoptic_channel*2, panoptic_channel)
        self.pre_fusion = nn.Linear(panoptic_channel*2, panoptic_channel)
        self.pre_fusion_0 = Linear_Residual(panoptic_channel)
        self.pre_fusion_1 = Linear_Residual(panoptic_channel)

        # after img & occ fusion
        self.mask_feat_extraction_0 = SparseConv3d_Residual(panoptic_channel, 3)
        self.mask_feat_extraction_1 = SparseConv3d_Residual(panoptic_channel, 3)
        self.mask_feat_extraction_2 = SparseConv3d_Residual(panoptic_channel, 3)

    # def feat_fusion_pre(self, feats_1x, feats_2x, feats_4x):
    #     feats_1x = self.self_fusion_1x(feats_1x)
    #     feats_2x = self.self_fusion_2x(feats_2x)
    #     feats_4x = self.self_fusion_4x(feats_4x)
    #
    #     feats_1x = F.interpolate(feats_1x, scale_factor=4, mode='bilinear')
    #     feats_2x = F.interpolate(feats_2x, scale_factor=2, mode='bilinear')
    #     feats_fusion = torch.concat([feats_1x, feats_2x, feats_4x], dim=1)
    #     feats_fusion = self.fusion_down(feats_fusion)  
    #
    #     # 图像特征 transfer 4✖3*3conv->9*9 感受野
    #     feats_fusion = self.post_fusion_1(feats_fusion)
    #     feats_fusion = self.post_fusion_2(feats_fusion)
    #
    #     return feats_fusion

    def img_feats_transfer(self, features_all, bs):
        # feats_1x = torch.stack([feat[2] for feat in features_all])
        # feats_2x = torch.stack([feat[1] for feat in features_all])
        feats_4x = torch.stack([feat[0] for feat in features_all])

        feats_fusion = []
        for batch in range(bs):
            feats_fusion.append([])
            # feats_fusion[-1] = self.feat_fusion_pre(feats_1x[:, batch], feats_2x[:, batch], feats_4x[:, batch])
            feats_fusion[-1] = feats_4x[:, batch]
            feats_fusion[-1] = feats_fusion[-1].unsqueeze(1)

        feats_fusion = torch.concat(feats_fusion, dim=1)

        return feats_fusion

    def fusion(self, img_feats, voxel_feats, coords, batch_size, spitial_shape):
        # occ_feats
        voxel_feats = self.occ2panoptic_0(voxel_feats)
        # voxel_feats = self.occ2panoptic_1(voxel_feats)

        # voxel_feats = self.occ2panoptic_conv0(x=voxel_feats, coords=coords, spitial_shape=spitial_shape, bs=batch_size)
        # voxel_feats = self.occ2panoptic_conv1(x=voxel_feats, coords=coords, spitial_shape=spitial_shape, bs=batch_size)
        # voxel_feats = self.occ2panoptic_conv2(x=voxel_feats, coords=coords, spitial_shape=spitial_shape, bs=batch_size)

        # img_feats
        img_feats = self.img2panoptic_0(img_feats)
        # img_feats = self.img2panoptic_1(img_feats)

        # fusion
        fusion_feats = torch.concat([img_feats, voxel_feats], dim=1)
        fusion_feats = self.pre_fusion(fusion_feats)
        fusion_feats = self.pre_fusion_0(fusion_feats)
        fusion_feats = self.pre_fusion_1(fusion_feats)

        return fusion_feats

    def generate_mask_features(self, panoptic_feats, coords_b, coords_xyz, batch_size, spitial_shape):
        coords = torch.cat([coords_b.unsqueeze(1), coords_xyz], dim=1)
        panoptic_feats = self.mask_feat_extraction_0(x=panoptic_feats, coords=coords, spitial_shape=spitial_shape, bs=batch_size)
        panoptic_feats = self.mask_feat_extraction_1(x=panoptic_feats, coords=coords, spitial_shape=spitial_shape, bs=batch_size)
        panoptic_feats = self.mask_feat_extraction_2(x=panoptic_feats, coords=coords, spitial_shape=spitial_shape, bs=batch_size)

        return panoptic_feats

