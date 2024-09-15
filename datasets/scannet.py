import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset


class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.absolute_source_path = '/home/zhouzhen/Project/ScanNetV2_reader'

        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}_1'.format(self.n_views)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.rgb_cashe = {}
        self.semantic_cashe = {}
        self.instance_cashe = {}
        self.max_cashe = 50  # 100 TODO: CPU memory increases sharply!

    def build_list(self):

        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)

        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)

        pose_path = "pose_" + "{:d}".format(vid)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', pose_path + '.txt'))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes_panoptic(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
                self.rgb_cashe = {}
                self.semantic_cashe = {}
                self.instance_cashe = {}

            full_tsdf_list = []
            full_rgb_list = []
            full_semantic_list = []
            full_insatnce_list = []

            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)), allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
                full_rgb = np.load(os.path.join(data_path, scene, 'full_rgb_layer{}.npz'.format(l)), allow_pickle=True)
                full_rgb_list.append(full_rgb.f.arr_0)
                full_semantic = np.load(os.path.join(data_path, scene, 'full_semantic_layer_interpolate{}.npz'.format(l)), allow_pickle=True)
                full_semantic_list.append(full_semantic.f.arr_0)
                full_instance = np.load(os.path.join(data_path, scene, 'full_instance_layer_interpolate{}.npz'.format(l)), allow_pickle=True)
                full_insatnce_list.append(full_instance.f.arr_0)

            self.tsdf_cashe[scene] = full_tsdf_list
            self.rgb_cashe[scene] = full_rgb_list
            self.semantic_cashe[scene] = full_semantic_list
            self.instance_cashe[scene] = full_insatnce_list

        return self.tsdf_cashe[scene], self.rgb_cashe[scene], self.semantic_cashe[scene], self.instance_cashe[scene]

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}

            full_tsdf_list = []

            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)), allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)

            self.tsdf_cashe[scene] = full_tsdf_list

        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        if self.mode == 'train':
            tsdf_list, rgb_list, semantic_list, instance_list = self.read_scene_volumes_panoptic(os.path.join(self.datapath, self.tsdf_file), meta['scene'])
        else:
            tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            # load images
            color_path = "color_" + "{:d}".format(vid)
            imgs.append(self.read_img(os.path.join(self.absolute_source_path, self.source_path, meta['scene'], 'color', color_path + '.jpg')))

            depth_path = "depth_" + "{:d}".format(vid)
            depth.append(self.read_depth(os.path.join(self.absolute_source_path, self.source_path, meta['scene'], 'depth', depth_path + '.png')))

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.absolute_source_path, self.source_path, meta['scene']), vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        if self.mode == 'train':
            items = {
                'imgs': imgs,
                'depth': depth,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'tsdf_list_full': tsdf_list,
                'rgb_list_full': rgb_list,
                'semantic_list_full': semantic_list,
                'instance_list_full': instance_list,
                'vol_origin': meta['vol_origin'],
                'scene': meta['scene'],
                'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
                'epoch': [self.epoch],
            }
        else:
            items = {
                'imgs': imgs,
                'depth': depth,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'tsdf_list_full': tsdf_list,
                'vol_origin': meta['vol_origin'],
                'scene': meta['scene'],
                'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
                'epoch': [self.epoch],
            }

        if self.transforms is not None:
            items = self.transforms(items)
        return items
