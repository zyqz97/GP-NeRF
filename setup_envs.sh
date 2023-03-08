# 1.Create new conda env
conda create -n gpnerf python=3.9
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install configargparse  PyYAML  opencv-python  tensorboard  tqdm  pyarrow  lpips  wandb  trimesh setuptools==56.1.0 pandas
#2.Build extension
cd /data/yuqi/code/GP-NeRF-private/mega_nerf/torch_ngp/shencoder
python setup.py install
cd ../raymarching
python setup.py install
cd ../gridencoder
python setup.py install
cd /data/yuqi/code/tiny-cuda-nn/bindings/torch
python setup.py install
cd /data/yuqi/code/GP-NeRF-private
