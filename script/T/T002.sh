#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=20GB
#PBS -l walltime=03:00:00
#PBS -l wd
#PBS -l storage=scratch/yp87

module load cuda/12.6.2

set -euo pipefail

source /scratch/yp87/sl5952/RASCAL/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 src/train_rascal.py --method rascal --dataset_name cifar10 --data_root ./data \
  --epochs 200 --batch_size 128 --model resnet50 --feat_dim 128 --temp 0.1 \
  --print_freq 100 \
  >> logs/rascal_cifar10.log 2>&1

python3 src/train_linear.py --ckpt ./save/rascal/rascal_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0/ckpt_last.pth \
  --dataset_name cifar10 --data_root ./data --train_split train --val_split test \
  --epochs 50 --batch_size 128 --model resnet50 \
  >> logs/rascal_cifar10_linear.log 2>&1