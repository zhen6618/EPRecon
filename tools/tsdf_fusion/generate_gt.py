import sys
sys.path.append('.')

import time
from tools.tsdf_fusion.fusion import *
import pickle
import argparse
from tqdm import tqdm
import ray
import torch.multiprocessing
from tools.simple_loader import *
import numpy as np
from scipy.sparse import csr_matrix

torch.multiprocessing.set_sharing_strategy('file_system')



def parse_args():
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf')
    parser.add_argument("--dataset", default='scannet')
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to raw dataset", default='datasets/scannet/')
    parser.add_argument("--save_name", metavar="DIR",
                        help="file name", default='all_tsdf_9')
    parser.add_argument('--test', action='store_true', default=True,
                        help='prepare the test set')
    parser.add_argument('--max_depth', default=3., type=float,
                        help='mask out large depth values since they are noisy')
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--margin', default=3, type=int)
    parser.add_argument('--voxel_size', default=0.04, type=float)

    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--min_angle', default=15, type=float)
    parser.add_argument('--min_distance', default=0.1, type=float, help='m')

    # ray multi processes
    parser.add_argument('--n_proc', type=int, default=1, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()


args = parse_args()
args.save_path = os.path.join(args.data_path, args.save_name)


def find_max_count_element(arr):
    """
    arr: [N, 1] class_id or instance_id
    """
    # 找到具有最多相同元素的元素
    unique_elements, counts = np.unique(arr, return_counts=True)
    max_count_element = unique_elements[np.argmax(counts)]

    # 如果每个元素都不一样，则随机选择一个元素
    if len(unique_elements) == len(arr):
        selected_element = np.random.choice(arr)
    else:
        # 否则，从具有最多相同元素的元素中随机选择一个
        max_count_indices = np.where(arr == max_count_element)[0]
        selected_element = np.random.choice(arr[max_count_indices])

    return selected_element

def find_average_count_element(arr):
    """
    arr: [N, 3] rgb
    """
    arr = np.mean(arr, axis=0)

    return arr


def integrate_semantic(xyz, rgb, semantic_labels, instance_labels, grid_shape):
    # 确保A, B, C是np.array类型
    xyz, rgb, semantic_labels, instance_labels = np.asarray(xyz), np.asarray(rgb), np.asarray(semantic_labels), np.asarray(instance_labels)

    # 生成一个唯一的一维索引
    # 假设grid_shape的最大值足够大，可以确保每个索引都是唯一的
    indices = xyz[:, 0] * (grid_shape[1] * grid_shape[2]) + xyz[:, 1] * grid_shape[2] + xyz[:, 2]

    # 对于RGB值，我们计算每个唯一索引对应的平均值
    rgb_vol = np.zeros(grid_shape + (3,))
    for i in range(3):  # 对于RGB的每个通道
        # 使用np.bincount计算加权平均，weights是当前通道的颜色值
        sums = np.bincount(indices.astype(int), weights=rgb[:, i], minlength=np.prod(grid_shape))
        counts = np.bincount(indices.astype(int), minlength=np.prod(grid_shape))
        averages = sums / np.maximum(counts, 1)  # 防止除以0
        rgb_vol[..., i] = averages.reshape(grid_shape)

    # 对于语义标签，我们找出每个位置最常见的标签
    # 将indices转换为整数
    indices = indices.astype(int)

    label_matrix = np.zeros((len(semantic_labels), np.max(semantic_labels) + 1), dtype=int)
    label_matrix[np.arange(len(semantic_labels)), semantic_labels[:, 0]] = 1  # 使用广播来设置标签的出现
    label_sums = np.zeros((np.prod(grid_shape), np.max(semantic_labels) + 1), dtype=int)
    np.add.at(label_sums, indices, label_matrix)

    # 找到每个格点频次最高的标签
    most_frequent_labels = np.argmax(label_sums, axis=1)
    semantic_vol = most_frequent_labels.reshape(grid_shape)

    # instance
    instance_matrix = np.zeros((len(instance_labels), np.max(instance_labels) + 1), dtype=int)
    instance_matrix[np.arange(len(instance_labels)), instance_labels[:, 0]] = 1  # 使用广播来设置标签的出现
    instance_sums = np.zeros((np.prod(grid_shape), np.max(instance_labels) + 1), dtype=int)
    np.add.at(instance_sums, indices, instance_matrix)

    # 找到每个格点频次最高的标签
    most_frequent_instances = np.argmax(instance_sums, axis=1)
    instance_vol = most_frequent_instances.reshape(grid_shape)

    return rgb_vol, semantic_vol, instance_vol


def save_tsdf_full(args, scene_path, cam_intr, depth_list, cam_pose_list, color_list, save_mesh=True):
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    vol_bnds = np.zeros((3, 2))

    n_imgs = len(depth_list.keys())
    if n_imgs > 200:
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_id = np.array(list(depth_list.keys()))[ind]
    else:
        image_id = depth_list.keys()
    for id in image_id:
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))  # vol_bnd_min (xyz bounds (min) in meters)
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))  # vol_bnd_max (xyz bounds (mmax) in meters)

    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol_list = []
    for l in range(args.num_layers):
        tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=args.voxel_size * 2 ** l, margin=args.margin))

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for id in depth_list.keys():
        if id % 100 == 0:
            print("{}: Fusing frame {}/{}".format(scene_path, str(id), str(n_imgs)))
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        if len(color_list) == 0:
            color_image = None
        else:
            color_image = color_list[id]

        # Integrate observation into voxel volume (assume color aligned with depth)
        for l in range(args.num_layers):
            tsdf_vol_list[l].integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    tsdf_info = {
        'vol_origin': tsdf_vol_list[0]._vol_origin,
        'voxel_size': tsdf_vol_list[0]._voxel_size,
    }
    tsdf_path = os.path.join(args.save_path, scene_path)
    if not os.path.exists(tsdf_path):
        os.makedirs(tsdf_path)

    with open(os.path.join(args.save_path, scene_path, 'tsdf_info.pkl'), 'wb') as f:
        pickle.dump(tsdf_info, f)

    for l in range(args.num_layers):
        tsdf_vol, color_vol, weight_vol = tsdf_vol_list[l].get_volume()  # tsdf_vol: [270, 265, 120]
        np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_tsdf_layer{}'.format(str(l))), tsdf_vol)

    '存储每个点的class_id, instance_id'
    if not args.test:
        for l in range(args.num_layers):
            vol_dim = tsdf_vol_list[l].get_vol()
            semantic_vol = np.zeros(vol_dim, dtype=np.int16)  # 0表示无类别
            instance_vol = np.zeros(vol_dim, dtype=np.int16)
            rgb_vol = np.tile(np.zeros(vol_dim, dtype=np.int16)[..., np.newaxis], 3)

            mesh_vertices = np.load(args.data_path.split('/scans')[0] + '/panoptic_info/' + scene_path + '_vert.npy')  # mesh_vertices: [N_voxels, 6] xyzrgb
            semantic_labels = np.load(args.data_path.split('/scans')[0] + '/panoptic_info/' + scene_path + '_sem_label.npy')  # semantic_labels: [N_voxels] class_id
            instance_labels = np.load(args.data_path.split('/scans')[0] + '/panoptic_info/' + scene_path + '_ins_label.npy')  # instance_labels: [N_voxels] instance_id
            rgb = mesh_vertices[:, 3:]  # [N_voxels, 3] rgb

            coords = mesh_vertices[:, :3]  # [N_voxels, 3] xyz
            coords = np.round((coords - np.tile(vol_bnds[:, 0], (mesh_vertices.shape[0], 1))) / (args.voxel_size * 2 ** l)).astype(int)
            coords[:, 0] = np.clip(coords[:, 0], 0, vol_dim[0]-1)  # X坐标限制
            coords[:, 1] = np.clip(coords[:, 1], 0, vol_dim[1]-1)  # Y坐标限制
            coords[:, 2] = np.clip(coords[:, 2], 0, vol_dim[2]-1)  # Z坐标限制

            rgb_vol, semantic_vol, instance_vol = integrate_semantic(coords, rgb, semantic_labels.reshape(-1, 1), instance_labels.reshape(-1, 1), (vol_dim[0], vol_dim[1], vol_dim[2]))
            # unique_coords = np.unique(coords, axis=0)
            #
            # for un, unique_coord in enumerate(unique_coords):
            #     if un % 1000 == 0:
            #         print(un, '/', unique_coords.shape[0])
            #
            #     unique_index = np.where(np.all(coords == unique_coord, axis=1))[0]
            #
            #     unique_rgb = rgb[unique_index]
            #     unique_semantic = semantic_labels[unique_index]
            #     unique_instance = instance_labels[unique_index]
            #
            #     selected_rgb = find_average_count_element(unique_rgb)
            #     selected_semantic = find_max_count_element(unique_semantic)
            #     selected_instance = find_max_count_element(unique_instance)
            #
            #     rgb_vol[unique_coord[0], unique_coord[1], unique_coord[2]] = np.int16(selected_rgb)
            #     semantic_vol[unique_coord[0], unique_coord[1], unique_coord[2]] = np.int16(selected_semantic)
            #     instance_vol[unique_coord[0], unique_coord[1], unique_coord[2]] = np.int16(selected_instance)

            np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_rgb_layer{}'.format(str(l))), rgb_vol)
            np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_semantic_layer{}'.format(str(l))), semantic_vol)
            np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_instance_layer{}'.format(str(l))), instance_vol)

    if save_mesh:
        for l in range(args.num_layers):
            print("Saving mesh to mesh{}.ply...".format(str(l)))
            verts, faces, norms, colors = tsdf_vol_list[l].get_mesh()

            meshwrite(os.path.join(args.save_path, scene_path, 'mesh_layer{}.ply'.format(str(l))), verts, faces, norms,
                      colors)

            # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
            # print("Saving point cloud to pc.ply...")
            # point_cloud = tsdf_vol_list[l].get_point_cloud()
            # pcwrite(os.path.join(args.save_path, scene_path, 'pc_layer{}.ply'.format(str(l))), point_cloud)


def save_fragment_pkl(args, scene, cam_intr, depth_list, cam_pose_list):
    fragments = []
    print('segment: process scene {}'.format(scene))

    # gather pose
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.inf
    vol_bnds[:, 1] = -np.inf

    all_ids = []
    ids = []
    all_bnds = []
    count = 0
    last_pose = None
    for id in depth_list.keys():
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        if count == 0:
            ids.append(id)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.inf
            vol_bnds[:, 1] = -np.inf
            last_pose = cam_pose
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                # Compute camera view frustum and extend convex hull
                view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    all_bnds.append(vol_bnds)
                    ids = []
                    count = 0

    with open(os.path.join(args.save_path, scene, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)

    # save fragments
    for i, bnds in enumerate(all_bnds):
        if not os.path.exists(os.path.join(args.save_path, scene, 'fragments', str(i))):
            os.makedirs(os.path.join(args.save_path, scene, 'fragments', str(i)))
        fragments.append({
            'scene': scene,
            'fragment_id': i,
            'image_ids': all_ids[i],
            'vol_origin': tsdf_info['vol_origin'],
            'voxel_size': tsdf_info['voxel_size'],
        })

    with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)


# @ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_with_single_worker(args, scannet_files):
    for scene in tqdm(scannet_files):
        if os.path.exists(os.path.join(args.save_path, scene, 'fragments.pkl')):
            continue
        print('read from disk')

        depth_all = {}
        cam_pose_all = {}
        color_all = {}

        if args.dataset == 'scannet':
            n_imgs = len(os.listdir(os.path.join(args.data_path, scene, 'color')))
            intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
            cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
            dataset = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
                                                 batch_sampler=None, num_workers=args.loader_num_workers)

        for id, (cam_pose, depth_im, _) in enumerate(dataloader):
            if id % 100 == 0:
                print("{}: read frame {}/{}".format(scene, str(id), str(n_imgs)))

            if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                continue
            depth_all.update({id: depth_im})
            cam_pose_all.update({id: cam_pose})
            # color_all.update({id: color_image})

        save_tsdf_full(args, scene, cam_intr, depth_all, cam_pose_all, color_all, save_mesh=True)
        save_fragment_pkl(args, scene, cam_intr, depth_all, cam_pose_all)


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def generate_pkl(args):
    all_scenes = sorted(os.listdir(args.save_path))
    # todo: fix for both train/val/test
    if not args.test:
        splits = ['train', 'val']
    else:
        splits = ['test']
    for split in splits:
        fragments = []
        # with open(os.path.join(args.save_path, 'splits', 'scannetv2_{}.txt'.format(split))) as f:
        print(args.data_path.split('/scans')[0])
        with open(os.path.join(args.data_path.split('/scans')[0], 'scannetv2_{}.txt'.format(split))) as f:
            split_files = f.readlines()
        for scene in all_scenes:
            if 'scene' not in scene:
                continue
            if scene + '\n' in split_files:
                with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'rb') as f:
                    frag_scene = pickle.load(f)
                fragments.extend(frag_scene)

        with open(os.path.join(args.save_path, 'fragments_{}.pkl'.format(split)), 'wb') as f:
            pickle.dump(fragments, f)


if __name__ == "__main__":

    import os
    print(os.getcwd())  # 输出当前debug所在的路径，可以指定为工程的主目录路径

    import subprocess

    os.environ['PATH'] += ':/usr/local/cuda/bin'  # CUDA 安装的实际路径

    result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print(result.stdout.decode('utf-8'))
    else:
        print("Error:", result.stderr.decode('utf-8'))

    all_proc = args.n_proc * args.n_gpu

    # ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    if args.dataset == 'scannet':
        if not args.test:
            args.data_path = os.path.join(args.data_path, 'scans')
        else:
            args.data_path = os.path.join(args.data_path, 'scans_test')
        files = sorted(os.listdir(args.data_path))
    else:
        raise NameError('error!')

    files = split_list(files, all_proc)

    # ray_worker_ids = []
    # for w_idx in range(all_proc):
    #     ray_worker_ids.append(process_with_single_worker.remote(args, files[w_idx]))

    # results = ray.get(ray_worker_ids)

    '不用分布式'
    results = process_with_single_worker(args, files[0])

    if args.dataset == 'scannet':
        generate_pkl(args)
