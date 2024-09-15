import torch
import torch.nn as nn
import psutil

from .backbone import MnasMulti
from .neucon_network import NeuConNet
from .gru_fusion import GRUFusion
from utils import tocuda

def print_system_memory(tag=""):
    mem = psutil.virtual_memory()
    # print(f"[{tag}] Total: {mem.total / 1024 ** 3:.2f} GB")
    # print(f"[{tag}] Available: {mem.available / 1024 ** 3:.2f} GB")
    print(f"[{tag}] Used: {mem.used / 1024 ** 3:.2f} GB")
    # print(f"[{tag}] Free: {mem.free / 1024 ** 3:.2f} GB")
    print("")


class NeuralRecon(nn.Module):
    def __init__(self, cfg):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
        self.backbone_occ_pano = MnasMulti(alpha)
        self.neucon_net = NeuConNet(cfg.MODEL)
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True, trianing=False)

        self.only_train_init = cfg.TRAIN.ONLY_INIT
        self.fuse_temporal = cfg.TRAIN.FUSE_TEMPORAL
        self.only_train_occ = cfg.TRAIN.ONLY_OCC

        self.init_overlap_count = 0

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs, save_mesh=False, training=True):
        inputs = tocuda(inputs)
        outputs = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        # image feature extraction
        # features: list9: list3: 1/4-[1, 24, 120, 160], 1/8-[1, 40, 60, 80], 1/16-[1, 80, 30, 40]
        features_backbone2d = [self.backbone2d(self.normalizer(img)) for img in imgs]
        features_backbone2d_occ_pano = [self.backbone_occ_pano(self.normalizer(img)) for img in imgs]

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features_backbone2d,
                                             features_backbone2d_occ_pano,
                                             inputs,
                                             outputs,
                                             only_train_init=self.only_train_init,
                                             only_train_occ=self.only_train_occ,
                                             init_overlap_count=self.init_overlap_count,)
        
        if self.only_train_init:
            self.init_overlap_count = outputs['init_overlap_count']
            # print('init_overlap_count: ', self.init_overlap_count)

        # fuse to global volume.
        if not training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh, panoptic_infos=outputs['panoptic_info'])

        # gather loss.
        print_loss = 'Loss: '
        for k, v in loss_dict.items():
            print_loss += f'{k}: {v} '

        weighted_loss = 0

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]

        loss_dict.update({'total_loss': weighted_loss})
        
        return outputs, loss_dict
