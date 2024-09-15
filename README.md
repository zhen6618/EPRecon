## EPRecon: An Efficient Framework for Real-Time Panoptic 3D Reconstruction from Monocular Video


## Installation
```
conda create -n EPRecon python=3.9
conda activate EPRecon

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

sudo apt-get install libsparsehash-dev
git clone -b v2.0.0 https://github.com/mit-han-lab/torchsparse.git
cd torchsparse
pip install tqdm
pip install .

git clone https://github.com/zhen6618/EPRecon.git
cd EPRecon

pip install -r requirements.txt
pip install sparsehash
pip install -U openmim
mim install mmcv-full
```   

## Dataset

1. Data Preperation for ScanNet
Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.

2. Data preparation script
   
For Geonetry Reconstruction:
```
# training/val split
python tools/tsdf_fusion/generate_gt.py --data_path datasets/scannet/ --save_name all_tsdf_9 --window_size 9
# test split
python tools/tsdf_fusion/generate_gt.py --test --data_path datasets/scannet/ --save_name all_tsdf_9 --window_size 9
```
For Panoptic Reconstruction:
```
python datasets/scannet/batch_load_scannet_data.py
python datasets/scannet/label_interpolate.py
```

