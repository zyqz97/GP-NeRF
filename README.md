# GP-NeRF

This repository contains the code needed to train GP-NeRF.

**Note:** This is a preliminary release and there may still be outstanding bugs.



## Setup
### Create new conda env ([pytorch-version](https://pytorch.org/get-started/previous-versions/), [CUDA](https://developer.nvidia.com/cuda-toolkit-archive))
```
conda create -n gpnerf python=3.9
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```


### Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```
tiny-cuda-nn$ cd bindings/torch
tiny-cuda-nn/bindings/torch$ python setup.py install
```


## Pretrained Models
comming soon.


## Datasets
Please refer to [Mega-NeRF](https://github.com/cmusatyalab/mega-nerf#data) for downloading the datasets.
```none
MegaNeRF
├── Mill19
│   ├── building
│   │   ├── building-pixsfm
│   ├── rubble
│   │   ├── rubble-pixsfm
│   ├── building_chunk-1 (auto processd by scripts when training if use "--dataset_type filesystem". You can choose another path to save.)
│   ├── rubble_chunk-1
├── Quad6k (the same as above)
│   ├── quad
├── UrbanScene3D (the same as above)
│   ├── residence
│   ├── sci-art
│   ├── campus
```

## Training

```
python gp_nerf/train.py --config_file configs/${DATASET_NAME}.yml --exp_name $EXP_PATH --dataset_path $DATASET_PATH --chunk_paths $CHUNK_PATH
```
At the first time of running, it takes some times to write the chunk into the disk.
## Evaluation

```
python gp_nerf/eval.py --config_file configs/${DATASET_NAME}.yaml  --exp_name $EXP_NAME --dataset_path $DATASET_PATH  --ckpt_path  $ckpt_path
```


## Acknowledgements

Large parts of this codebase are based on existing work in the [Mega-NeRF](https://github.com/cmusatyalab/mega-nerf) and [torch-ngp](https://github.com/ashawkey/torch-ngp) repositories.
