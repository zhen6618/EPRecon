import argparse
from concurrent.futures import process
import os, sys
from tqdm import tqdm
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager

from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--scans_folder', default='/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data')
parser.add_argument('--scan_list_file', default='/root/paddlejob/workspace/env_run/zhouzhen05/Code/DRecon/datasets/scannet/meta_data/scannetv2_train.txt')
parser.add_argument('--single_debug_scan_id', required=False, default=None, help='single scan to debug')
parser.add_argument('--output_path', default='/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/Scannet_Reader/scans')
parser.add_argument('--export_depth_images', default=True)
parser.add_argument('--export_color_images', default=True)
parser.add_argument('--export_poses', default=True)
parser.add_argument('--export_intrinsics', default=True)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--rgb_resize', nargs='+', type=int, default=None, help='width height')
parser.add_argument('--depth_resize', nargs='+', type=int, default=None, help='width height')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()
print(opt)

def process_scan(opt, scan_job, count=None, progress=None):
  filename = scan_job[0]
  output_path = scan_job[1]
  scan_name = scan_job[2]

  if not os.path.exists(output_path):
      os.makedirs(output_path)
  # load the data
  sys.stdout.write('loading %s...' % opt.scans_folder)
  sd = SensorData(filename)
  sys.stdout.write('loaded!\n')
  
  opt.export_depth_images, opt.export_color_images, opt.export_poses, opt.export_intrinsics = True, True, True, True
  if opt.export_depth_images:
    sd.export_depth_images(os.path.join(output_path, 'depth'), image_size=opt.depth_resize)
  if opt.export_color_images:
    sd.export_color_images(os.path.join(output_path, 'color'), image_size=opt.rgb_resize)
  if opt.export_poses:
    sd.export_poses(os.path.join(output_path, 'pose'))
  if opt.export_intrinsics:
    sd.export_intrinsics(output_path, scan_name)

  if progress is not None:
    progress.value += 1
    print(f"Completed scan {filename}, {progress.value} of total {count}.")

def main():


  if opt.single_debug_scan_id is not None:
    scans = [opt.single_debug_scan_id]
  else:
    f = open(opt.scan_list_file, "r")
    scans = f.readlines()
    scans = [scan.strip() for scan in scans]
  
  # input_files = [os.path.join(opt.scans_folder, f"{scan}/{scan}.sens") for 
  #                                       scan in scans]
  input_files = [os.path.join(opt.scans_folder, f"{scan}.sens") for scan in scans]

  output_dirs = [os.path.join(opt.output_path, scan) for scan in scans]

  scan_jobs = list(zip(input_files, output_dirs, scans))

  if opt.num_workers == 1:
    for scan_job in tqdm(scan_jobs):
      process_scan(opt, scan_job)
  else:

    pool = Pool(opt.num_workers)
    manager = Manager()

    count = len(scan_jobs)
    progress = manager.Value('i', 0)


    pool.map(
          partial(
              process_scan,
              opt,
              count=count,
              progress=progress
          ),
          scan_jobs,
    )

if __name__ == '__main__':
    main()
