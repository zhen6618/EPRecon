import numpy as np
import plyfile
from scipy.spatial import cKDTree
import os
from collections import Counter

def read_ply(ply_file):
    plydata = plyfile.PlyData.read(ply_file)
    vertices = np.vstack([plydata['vertex'][dim] for dim in ('x', 'y', 'z')]).T
    return vertices

def generate_semantic_instance(scene_name):
    pred_file_all = f"results/scene_scannet_checkpoints_fusion_eval_99/{scene_name}.npz"
    pred_all = np.load(pred_file_all)

    pred_origin = pred_all['origin']
    pred_voxel_size = pred_all['voxel_size']
    pred_tsdf_volume = pred_all['tsdf']
    pred_semantic_volume = pred_all['semantic']
    pred_instance_volume = pred_all['instance']

    # transfer predictions to world coordinate
    pred_shape = pred_semantic_volume.shape
    indices = np.indices((pred_shape[0], pred_shape[1], pred_shape[2]))
    coords_pred = np.stack(indices, axis=-1).reshape(-1, 3)
    coords_pred = coords_pred * pred_voxel_size + pred_origin  # voxel grid coordinates to world coordinates

    semantic_pred = pred_semantic_volume.reshape(-1)
    instance_pred = pred_instance_volume.reshape(-1)

    # Semantic Mapping
    mapping = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    semantic_pred = mapping[semantic_pred.astype(int)]

    # eliminate 0 class
    semantic_pred_non_zero = semantic_pred != 0
    coords_pred = coords_pred[semantic_pred_non_zero]
    semantic_pred = semantic_pred[semantic_pred_non_zero]
    instance_pred = instance_pred[semantic_pred_non_zero]

    # Building a KDTree
    kdtree = cKDTree(coords_pred)

    ref_file_path = f'/home/zhouzhen/Project/ScanNetV2_ply/{scene_name}_vh_clean_2.ply'
    ply_vertices = read_ply(ref_file_path)

    # Find the nearest neighbors of each PLY vertex
    distances, indices = kdtree.query(ply_vertices)

    # Mapping Tags
    mapped_semantic = semantic_pred[indices]
    mapped_instance = instance_pred[indices]

    # save semantic txt
    os.makedirs('semantic', exist_ok=True)
    output_file = 'semantic/' + scene_name + '.txt'
    np.savetxt(output_file, mapped_semantic, fmt='%d')

    # save instance txt
    os.makedirs('instance', exist_ok=True)
    os.makedirs('instance/predicted_masks', exist_ok=True)
    unique_instance_ids = np.unique(mapped_instance).astype(int)  # Get all instance IDs

    # Traverse each instance ID and create a corresponding mask file
    for i, instance_id in enumerate(unique_instance_ids):
        mask_filename = f'instance/predicted_masks/{scene_name}_{i:03d}.txt'
        instance_mask = (mapped_instance == instance_id).astype(int)  # Generate a mask, the vertices belonging to this instance are 1, and the rest are 0
        np.savetxt(mask_filename, instance_mask, fmt='%d')

    # Generate the main prediction file
    prediction_file = f'instance/{scene_name}.txt'
    with open(prediction_file, 'w') as f:
        for i, instance_id in enumerate(unique_instance_ids):
            mask_filename = f'predicted_masks/{scene_name}_{i:03d}.txt'

            counter = Counter(mapped_semantic[mapped_instance == instance_id])
            class_label_id = counter.most_common(1)[0][0]  # Find the element with the most occurrences

            confidence_score = 1.0  # Example confidence score, which can be adjusted according to the specific model
            f.write(f'{mask_filename} {class_label_id} {confidence_score:.4f}\n')


if __name__ == "__main__":
    file_path = 'datasets/scannet/scannetv2_test.txt'

    with open(file_path, 'r', encoding='utf-8') as file:
        scene_names = [line.strip() for line in file]

    for scene_name in scene_names:
        print(scene_name)
        generate_semantic_instance(scene_name)
        
    # zip
    # zip -r instance.zip instance

