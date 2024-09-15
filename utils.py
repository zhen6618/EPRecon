import os
import torch
import trimesh
import numpy as np
import torchvision.utils as vutils
from skimage import measure
from loguru import logger
from tools.render import Visualizer
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from PIL import Image


# print arguments
def print_args(args):
    logger.info("################################  args  ################################")
    for k, v in args.__dict__.items():
        logger.info("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    logger.info("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        if len(vars.shape) == 0:
            return vars.data.item()
        else:
            return [v.data.item() for v in vars]
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def sparse_to_dense_torch_batch(locs, values, dim, default_val):
    dense = torch.full([dim[0], dim[1], dim[2], dim[3]], default_val, dtype=values.dtype, device=locs.device)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3]] = values
    return dense


def sparse_to_dense_torch(locs, values, dim, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2]], default_val, dtype=values.dtype, device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_channel(locs, values, dim, c, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2], c], default_val, dtype=values.dtype, device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_np(locs, values, dim, default_val):
    dense = np.zeros([dim[0], dim[1], dim[2]], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


class SaveScene(object):
    def __init__(self, cfg):
        self.cfg = cfg
        log_dir = cfg.LOGDIR.split('/')[-1]
        self.log_dir = os.path.join('results', 'scene_' + cfg.DATASET + '_' + log_dir)
        self.scene_name = None
        self.global_origin = None
        self.tsdf_volume = []  # not used during inference.
        self.weight_volume = []

        self.coords = None

        self.keyframe_id = None

        if cfg.VIS_INCREMENTAL:
            self.vis = Visualizer()

    def close(self):
        self.vis.close()
        cv2.destroyAllWindows()

    def reset(self):
        self.keyframe_id = 0
        self.tsdf_volume = []
        self.weight_volume = []

        # self.coords = coordinates(np.array([416, 416, 128])).float()

        # for scale in range(self.cfg.MODEL.N_LAYER):
        #     s = 2 ** (self.cfg.MODEL.N_LAYER - scale - 1)
        #     dim = tuple(np.array([416, 416, 128]) // s)
        #     self.tsdf_volume.append(torch.ones(dim).cuda())
        #     self.weight_volume.append(torch.zeros(dim).cuda())

    @staticmethod
    def tsdf2mesh(voxel_size, origin, tsdf_vol):
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
        return mesh

    @staticmethod
    def tsdf_panoptic2mesh(voxel_size, origin, tsdf_vol, semantic_vol, instance_vol):
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)  # level: The scalar value of the isosurface. The function will extract the surface corresponding to this scalar value.

        # semantic & instance
        rounded_verts = np.round(verts).astype(int)  # Round vertex coordinates to the nearest integer for use as indices
        rounded_verts = np.clip(rounded_verts, [0, 0, 0], np.array(semantic_vol.shape) - 1)  # Fix out-of-bounds indexes
        semantics = semantic_vol[rounded_verts[:, 0], rounded_verts[:, 1], rounded_verts[:, 2]]
        instances = instance_vol[rounded_verts[:, 0], rounded_verts[:, 1], rounded_verts[:, 2]]

        # mesh
        verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)

        color_palette = [
            [255, 192, 203], [128, 128, 128], [144, 238, 144], [0, 0, 255], [255, 255, 0], [0, 255, 255],
            [0, 128, 255], [128, 0, 255], [255, 0, 128], [255, 0, 0], [255, 255, 255],
            [255, 192, 203], [75, 0, 130], [255, 165, 0], [0, 100, 0], [255, 20, 147],
            [100, 149, 237], [255, 105, 180], [205, 92, 92], [186, 85, 211], [124, 252, 0],
            [70, 130, 180], [255, 215, 0], [0, 255, 255], [255, 69, 0], [138, 43, 226],
            [255, 105, 180], [70, 130, 180], [255, 192, 203], [219, 112, 147], [128, 128, 0],
            [255, 105, 180], [255, 20, 147], [255, 99, 71], [255, 69, 0], [255, 215, 0],
            [255, 182, 193], [0, 255, 0], [0, 255, 127], [34, 139, 34], [255, 240, 245],
            [255, 0, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
            [0, 128, 128], [128, 0, 128], [255, 128, 0], [128, 255, 0], [0, 255, 128],
        ]  # RGB
        color_palette = np.array(color_palette)
        num_colors = len(color_palette)

        # Map semantic labels to predefined colors
        semantic_colors = color_palette[semantics.astype(int)]
        instances_mapped = instances % num_colors
        instance_colors = color_palette[instances_mapped.astype(int)]

        # Copy the mesh and assign vertex colors
        mesh_semantic = deepcopy(mesh)
        mesh_instance = deepcopy(mesh)

        # Set semantic color
        mesh_semantic.visual.vertex_colors = semantic_colors
        # Set instance color
        mesh_instance.visual.vertex_colors = instance_colors

        # # semantic
        # mesh_semantic = deepcopy(mesh)
        # cmap = plt.get_cmap('jet')
        # norm = Normalize(vmin=np.min(semantics), vmax=np.max(semantics))  # [0, 1]
        # colors = cmap(norm(semantics))
        # mesh_semantic.visual.vertex_colors = colors[:, :3] * 255
        #
        # # instance
        # mesh_instance = deepcopy(mesh)
        # cmap = plt.get_cmap('jet')
        # norm = Normalize(vmin=np.min(instances), vmax=np.max(instances))
        # colors = cmap(norm(instances))
        # mesh_instance.visual.vertex_colors = colors[:, :3] * 255

        return mesh, mesh_semantic, mesh_instance

    def vis_incremental(self, epoch_idx, batch_idx, imgs, outputs):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # vis
            key_frames = []
            for img in imgs[::3]:
                img = img.permute(1, 2, 0)
                img = img[:, :, [2, 1, 0]]
                img = img.data.cpu().numpy()
                img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                key_frames.append(img)
            key_frames = np.concatenate(key_frames, axis=0)
            # cv2.imshow('Selected Keyframes', key_frames / 255)
            # cv2.waitKey(1)

            # vis mesh
            save_path = '{}_fusion_eval_{}'.format(self.log_dir, epoch_idx)
            mesh.export(os.path.join(save_path, '{}.ply'.format(self.scene_name)))
            self.vis.vis_mesh(mesh)

    def save_incremental(self, epoch_idx, batch_idx, imgs, outputs):
        save_path = os.path.join('incremental_' + self.log_dir + '_' + str(epoch_idx), self.scene_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        mesh_path = os.path.join(save_path, 'mesh')
        if not os.path.exists(mesh_path):
            os.makedirs(mesh_path)

        mesh_semantic_path = os.path.join(save_path, 'mesh_semantic')
        if not os.path.exists(mesh_semantic_path):
            os.makedirs(mesh_semantic_path)

        mesh_instance_path = os.path.join(save_path, 'mesh_instance')
        if not os.path.exists(mesh_instance_path):
            os.makedirs(mesh_instance_path)

        image_path = os.path.join(save_path, 'mesh_image')
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        for i, img in enumerate(imgs):
            img = img.permute(1, 2, 0)
            img = img.data.cpu().numpy().astype(np.uint8)
            image = Image.fromarray(img)
            image.save(os.path.join(image_path, 'image_{}_{}.png'.format(self.keyframe_id, i)))

        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        instance_volume = outputs['scene_instance'][batch_idx].data.cpu().numpy()
        semantic_volume = outputs['scene_semantic'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh, mesh_semantic, mesh_instance = self.tsdf_panoptic2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume, semantic_volume, instance_volume)
            # save
            mesh.export(os.path.join(mesh_path, 'mesh_{}.ply'.format(self.keyframe_id)))
            mesh_semantic.export(os.path.join(mesh_semantic_path, 'mesh_semantic_{}.ply'.format(self.keyframe_id)))
            mesh_instance.export(os.path.join(mesh_instance_path, 'mesh_instance_{}.ply'.format(self.keyframe_id)))

    def save_scene_eval(self, epoch, outputs, batch_idx=0):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        instance_volume = outputs['scene_instance'][batch_idx].data.cpu().numpy()
        semantic_volume = outputs['scene_semantic'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()

        if (tsdf_volume == 1).all():
            logger.warning('No valid data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            # mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            mesh, mesh_semantic, mesh_instance = self.tsdf_panoptic2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume, semantic_volume, instance_volume)

            # save tsdf volume for atlas evaluation
            data = {'origin': origin,
                    'voxel_size': self.cfg.MODEL.VOXEL_SIZE,
                    'tsdf': tsdf_volume,
                    'semantic': semantic_volume,
                    'instance': instance_volume,
                    }
            save_path = '{}_fusion_eval_{}'.format(self.log_dir, epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez_compressed(os.path.join(save_path, '{}.npz'.format(self.scene_name)), **data)
            mesh.export(os.path.join(save_path, '{}.ply'.format(self.scene_name)))
            mesh_semantic.export(os.path.join(save_path, 'mesh_semantic_{}.ply'.format(self.scene_name)))
            mesh_instance.export(os.path.join(save_path, 'mesh_instance_{}.ply'.format(self.scene_name)))

    def __call__(self, outputs, inputs, epoch_idx):
        # no scene saved, skip
        if "scene_name" not in outputs.keys():
            return

        batch_size = len(outputs['scene_name'])
        for i in range(batch_size):
            scene = outputs['scene_name'][i]
            self.scene_name = scene.replace('/', '-')
            
            if self.cfg.SAVE_INCREMENTAL:
                self.save_incremental(epoch_idx, i, inputs['imgs'][i], outputs)

            # if self.cfg.VIS_INCREMENTAL:
            #     self.vis_incremental(epoch_idx, i, inputs['imgs'][i], outputs)

            # if self.cfg.VIS_INCREMENTAL:
            #     self.close()

            if self.cfg.SAVE_SCENE_MESH:
                self.save_scene_eval(epoch_idx, outputs, i)
