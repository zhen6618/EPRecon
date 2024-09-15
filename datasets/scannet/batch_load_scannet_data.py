import argparse
import datetime
import os
from os import path as osp

import numpy as np
from load_scannet_data import export

# scannet provides 607 objects, which are summarized into 40 categories in nyuv2-40;
# many current instance segmentation tasks are processed according to the categories in Nyuv2,
# where [1, 2] is [wall, floor], and the remaining 18 categories are instance classes (sequence numbers are as follows)

# nyuv2-40:
# 1-wall  2-floor  3-cabinet  4-bed  5-chair
# 6-sofa  7-table  8-door  9-window  10-bookshelf
# 11-picture  12-counter  13-blinds  14-desk  15-shelves
# 16-curtain  17-dresser  18-pillow  19-mirror  20-floor mat
# 21-clothes  22-ceiling  23-books  24-refrigerator  25-television
# 26-paper  27-towel  28-shower curtain  29-box  30-whiteboard
# 31-person  32-night stand  33-toilet  34-sink  35-lamp
# 36-bathtub  37-bag  38-otherstructure  39-otherfurniture  40-otherprop

# nyuv2-20:
# 1-wall  2-floor  3-cabinet  4-bed  5-chair
# 6-sofa  7-table  8-door  9-window  10-bookshelf
# 11-picture  12-counter  14-desk  16-curtain 24-refridgerator
# 28-shower curtain  33-toilet  34-sink  36-bathtub  39-otherfurniture

# When reading data, if 0 appears, it means there is no category
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])  # for instance segmentation task
DONOTCARE_CLASS_IDS = np.array([])

def reassign_ids(A, B):
    new_A = np.zeros_like(A)

    # For labels of 0, 1, and 2, directly set the corresponding instance id to its label value
    for label in [0, 1, 2]:  # 0 (no category), 1 (wall background category), 2 (floor background category)
        mask = B == label
        new_A[mask] = label

    mask = (B != 0) & (B != 1) & (B != 2)
    unique_ids = np.unique(A[mask])

    next_id = 3
    for id in unique_ids:
        id_mask = A == id
        new_A[id_mask & mask] = next_id
        next_id += 1

    return new_A

def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    label_map_file,
                    scannet_dir,
                    test_mode=False):
    mesh_file = osp.join(scannet_dir, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '.aggregation.json')
    seg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '_vh_clean_2.0.010000.segs.json')
    # includes axisAlignment info for the train set scans.
    'mesh_vertices: '
    'mesh_vertices: [N_voxels, 6] xyzrgb'
    'semantic_labels: [N_voxels] class_id'
    'instance_labels: [N_voxels] instance_id'
    'unaligned_bboxes: [N_instance, 7] xyzwhl+class_id'
    'aligned_bboxes: [N_instance, 7] xyzwhl+class_id'
    'instance2semantic: [N_instance] corresponding class_id for each instance_id'
    'axis_align_matrix: [4, 4] axis-align transformation matrix'
    meta_file = osp.join(scannet_dir, scan_name, f'{scan_name}.txt')
    mesh_vertices, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, instance2semantic, axis_align_matrix = export(
            mesh_file, agg_file, seg_file, meta_file, label_map_file, None,
            test_mode)

    if not test_mode:
        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')

        bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f'Num of care instances: {unaligned_bboxes.shape[0]}')

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]
    # 0 (no category), 1 (wall background category), 2 (floor background category) are fixed, the rest of the instances remain unchanged, and the ids are reordered
    instance_labels = reassign_ids(instance_labels, semantic_labels)

    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)
    if not test_mode:
        np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
        np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
        np.save(f'{output_filename_prefix}_unaligned_bbox.npy', unaligned_bboxes)
        np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)
        np.save(f'{output_filename_prefix}_axis_align_matrix.npy', axis_align_matrix)


def batch_export(max_num_point,
                 output_folder,
                 scan_names_file,
                 label_map_file,
                 scannet_dir,
                 test_mode=False):
    if test_mode and not os.path.exists(scannet_dir):
        # test data preparation is optional
        return
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    scan_names = [line.rstrip() for line in open(scan_names_file)]
    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        try:
            export_one_scan(scan_name, output_filename_prefix, max_num_point, label_map_file, scannet_dir, test_mode)
        except Exception:
            print(f'Failed export scan: {scan_name}')
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='panoptic_info_debug',
        help='output folder of the result.')
    parser.add_argument(
        '--train_scannet_dir', default='scans', help='scannet data directory.')
    parser.add_argument(
        '--test_scannet_dir',
        default='scans_test',
        help='scannet data directory.')
    parser.add_argument(
        '--label_map_file',
        default='meta_data/scannetv2-labels.combined.tsv',
        help='The path of label map file.')
    parser.add_argument(
        '--train_scan_names_file',
        default='meta_data/scannet_train.txt',
        help='The path of the file that stores the scan names.')
    parser.add_argument(
        '--test_scan_names_file',
        default='meta_data/scannetv2_test.txt',
        help='The path of the file that stores the scan names.')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.train_scan_names_file,
        args.label_map_file,
        args.train_scannet_dir,
        test_mode=False)

    batch_export(
        args.max_num_point,
        args.output_folder,
        args.test_scan_names_file,
        args.label_map_file,
        args.test_scannet_dir,
        test_mode=True)


if __name__ == '__main__':
    main()
