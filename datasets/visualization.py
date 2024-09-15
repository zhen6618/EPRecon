import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D


def plot_voxels(voxels, title="Voxel Grid"):
    grid = pv.UniformGrid()
    grid.dimensions = np.array(voxels.shape) + 1

    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    grid.cell_data["values"] = voxels.flatten(order="F")  # Flatten the array!

    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="values", opacity='sigmoid', show_scalar_bar=False)
    plotter.set_background("white")
    plotter.camera_position = 'iso'
    plotter.add_title(title)
    plotter.show()


def visualize_mesh(input_A=None, input_B=None, type='xyz', use_path=False):

    # A = np.load('datasets/scannet/panoptic_info/scene0000_00_vert.npy')  xyz(N, 3) or xyzrgb(N, 6)
    # B = np.load('datasets/scannet/panoptic_info/scene0000_00_sem_label.npy')  class_id(N,)
    # C = np.load('datasets/scannet/panoptic_info/scene0000_00_ins_label.npy')  instance_id(N,)

    if use_path:
        A = np.load(input_A)
    else:
        A = input_A
    N = A.shape[0]
    A = A.reshape(N, -1)
    if type == 'xyz':
        points = pv.PolyData(A[:, :3])

        gray_color = np.array([0.5, 0.5, 0.5])
        points.point_data['gray_colors'] = np.tile(gray_color, (N, 1))

        plotter = pv.Plotter()
        plotter.add_points(points, scalars='gray_colors', rgb=True, point_size=5)

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        plotter.view_vector([-1, -1, 0])
        plotter.reset_camera()

        plotter.show(title='PointCloud Gray Visualization')

    elif type == 'rgb':  # 0-255
        rgb_values = A[:, 3:]
        non_zero_indices = ~np.all(rgb_values == 0, axis=1)
        A = A[non_zero_indices]

        points_rgb = pv.PolyData(A[:, :3])
        points_rgb['rgb'] = A[:, 3:6] / 255

        plotter = pv.Plotter()
        plotter.add_points(points_rgb, rgb=True, point_size=5)

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        plotter.view_vector([-1, -1, 0])
        plotter.reset_camera()

        plotter.show(title='Point Cloud with RGB Colors')

    elif type == 'semantic':
        if use_path:
            B = np.load(input_B)
        else:
            B = input_B
        B = B.reshape(N, -1)

        A = A[B.reshape(-1) > 0]
        B = B[B.reshape(-1) > 0]

        # manual_colors = np.array([
        #     [0, 0, 1],
        #     [1, 0, 0],
        #     [0, 1, 0]
        # ])

        # num_classes = 51
        # colors = plt.get_cmap('Set3', num_classes)
        # generated_colors = [colors(i / num_classes) for i in range(num_classes)]
        # generated_colors = np.array(generated_colors)[:, :3]
        #
        # semantic_colors = np.vstack((manual_colors, generated_colors[3:num_classes]))
        #
        # semantic_class_colors = np.array([semantic_colors[i[0]] for i in B])

        semantic_class_colors = np.random.rand(51, 3)
        semantic_class_colors = semantic_class_colors[B.flatten()]

        points_semantic = pv.PolyData(A[:, :3])
        points_semantic['semantic_colors'] = semantic_class_colors

        plotter = pv.Plotter()
        plotter.add_points(points_semantic, rgb=True, point_size=5)

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        plotter.view_vector([-1, -1, 0])
        plotter.reset_camera()

        plotter.show(title='Point Cloud with Semantic Labels')

    elif type == 'instance':
        if use_path:
            B = np.load(input_B)
        else:
            B = input_B
        B = B.reshape(N, -1)

        A = A[B.reshape(-1) != 0]
        B = B[B.reshape(-1) != 0]

        points = pv.PolyData(A[:, :3])

        instance_colors = np.random.rand(np.max(B)+1, 3)
        instance_colors = instance_colors[B.flatten()]

        points.point_data['instance_colors'] = instance_colors

        plotter = pv.Plotter()
        plotter.add_points(points, scalars='instance_colors', rgb=True, point_size=5)

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        plotter.view_vector([-1, -1, 0])
        plotter.reset_camera()

        plotter.show(title='Instance Segmentation Result')

    elif type == 'tsdf':
        if use_path:
            B = np.load(input_B)
        else:
            B = input_B
        B = B.reshape(N, -1)

        A = A[B.reshape(-1) > 0]
        B = B[B.reshape(-1) > 0]

        # normalized_tsdf = (B - np.min(B)) / (np.max(B) - np.min(B)) * 2 - 1
        normalized_tsdf = B

        grid = pv.PolyData(A[:, :3])

        grid.point_data['tsdf'] = normalized_tsdf.flatten()

        plotter = pv.Plotter()
        plotter.add_mesh(grid, scalars='tsdf', cmap='plasma')

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        plotter.view_vector([-1, -1, 0])
        plotter.reset_camera()

        plotter.show(title='TSDF Visualization')

    else:
        pass


if __name__ == '__main__':
    path_A = 'datasets/scannet/panoptic_info/scene0000_00_vert.npy'
    path_B = 'datasets/scannet/panoptic_info/scene0000_00_sem_label.npy'
    path_C = 'datasets/scannet/panoptic_info/scene0000_00_ins_label.npy'

    # visualize_mesh(path_A, type='xyz', use_path=True)
    # visualize_mesh(path_A, type='rgb', use_path=True)
    # visualize_mesh(path_A, path_B, type='semantic', use_path=True)
    # visualize_mesh(path_A, path_C, type='instance', use_path=True)
    visualize_mesh(path_A, path_B, type='tsdf', use_path=True)


