import numpy as np
import os
from scipy.interpolate import NearestNDInterpolator
import time

def main():
    root_path = 'datasets/scannet/all_tsdf_9'
    folder_names = [folder for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]

    count = 0
    for folder_name in folder_names:
        print(count / len(folder_names))
        count += 1
        folder_name = os.path.join(root_path, folder_name)
        if os.path.exists(os.path.join(folder_name, 'full_instance_layer0.npz')):
            for i in range(3):
                instance_name = os.path.join(folder_name, 'full_instance_layer' + str(i) + '.npz')
                semantic_name = os.path.join(folder_name, 'full_semantic_layer' + str(i) + '.npz')

                instance = np.load(instance_name, allow_pickle=True)
                instance = instance.f.arr_0
                semantic = np.load(semantic_name, allow_pickle=True)
                semantic = semantic.f.arr_0

                '--- Nearest neighbor interpolation ---'
                non_zero_indices = np.where(instance != 0)
                non_zero_values = instance[non_zero_indices]
                non_zero_coords = np.transpose(non_zero_indices)
                interpolator = NearestNDInterpolator(non_zero_coords, non_zero_values)

                grid_coords = np.indices(instance.shape).reshape(instance.ndim, -1).T
                interpolated_values = interpolator(grid_coords).reshape(instance.shape)

                new_instance_name = os.path.join(folder_name, 'full_instance_layer_interpolate' + str(i) + '.npz')
                np.savez_compressed(new_instance_name, interpolated_values)
                
                t1 = time.time()
                non_zero_indices = np.where(semantic != 0)
                non_zero_values = semantic[non_zero_indices]
                non_zero_coords = np.transpose(non_zero_indices)
                interpolator = NearestNDInterpolator(non_zero_coords, non_zero_values)

                grid_coords = np.indices(semantic.shape).reshape(semantic.ndim, -1).T
                interpolated_values = interpolator(grid_coords).reshape(semantic.shape)
                print(time.time() - t1)

                new_semantic_name = os.path.join(folder_name, 'full_semantic_layer_interpolate' + str(i) + '.npz')
                np.savez_compressed(new_semantic_name, interpolated_values)


if __name__ == '__main__':
    main()
