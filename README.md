
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
