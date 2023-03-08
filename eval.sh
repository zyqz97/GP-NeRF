#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4

exp_name=logs/eval
ckpt_path=   # give the checkpoint path  
dataset1='Mill19'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='building' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
python gp_nerf/eval.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --exp_name  $exp_name    --ckpt_path  $ckpt_path













