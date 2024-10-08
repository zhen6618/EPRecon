## EPRecon: An Efficient Framework for Real-Time Panoptic 3D Reconstruction from Monocular Video

<p align="center">
  <img src="https://github.com/zhen6618/EPRecon/blob/main/demo/demo.gif" alt="Alt Text">
</p>

<div align=center>
<img src="https://github.com/zhen6618/EPRecon/blob/main/demo/Overview.png" width="1000px"> 
</div>

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

1. Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.
```
python datasets/scannet/download_scannet.py
```
2. Generate depth, color, pose, intrinsics from .sens file (change your file path)
```
python datasets/scannet/reader.py
```
Expected directory structure of ScanNet can refer to [NeuralRecon](https://github.com/zju3dv/NeuralRecon)

3. Extract instance-level semantic labels (change your file path).
```
python datasets/scannet/batch_load_scannet_data.py
```
4. Label generation for panoptic reconstruction (change your file path):
```
# training/val split
python tools/tsdf_fusion/generate_gt.py --data_path datasets/scannet/ --save_name all_tsdf_9 --window_size 9
# test split
python tools/tsdf_fusion/generate_gt.py --test --data_path datasets/scannet/ --save_name all_tsdf_9 --window_size 9
```
5. Panoptic label interpolation (change your file path):
```
python datasets/scannet/label_interpolate.py
```

## Training
```
python main.py --cfg ./config/train.yaml
```

## Testing
```
python main.py --cfg ./config/test.yaml
```

## Generate Results for Evaluation
```
python tools/generate_semantic_instance.py
```

## Citation
```
@InProceedings{zhou2024epreconefficientframeworkrealtime,
      title={EPRecon: An Efficient Framework for Real-Time Panoptic 3D Reconstruction from Monocular Video}, 
      author={Zhen Zhou and Yunkai Ma and Junfeng Fan and Shaolin Zhang and Fengshui Jing and Min Tan},
      year={2024},
      booktitle={arXiv preprint arXiv:2409.01807},
}
```

