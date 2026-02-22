# RASCAL

This repo contains runnable scripts for:

- **Contrastive pretraining** with **SupCon** or **RASCAL** (drop-in replacement)
- **Linear probe** evaluation (Top-1 / Top-5)

All training code lives in `src/`.

## Setup

Install Python deps (PyTorch install is environment-specific, so install that first):

```bash
pip install -r requirements.txt
```

Notes:

- Backbones are created via `timm` (see `src/resnet_big.py`).
- Datasets are loaded via `src/ufgvc.py` (`UFGVCDataset`) and can auto-download parquet files.

## 1) Contrastive pretraining (SupCon vs RASCAL)

Run from the repo root.

### SupCon pretrain

```bash
python src/train_rascal.py --method supcon --dataset_name cifar10 --data_root ./data \
	--epochs 200 --batch_size 128 --model resnet50 --feat_dim 128 --temp 0.1
```

### RASCAL pretrain

```bash
python src/train_rascal.py --method rascal --dataset_name cifar10 --data_root ./data \
	--epochs 200 --batch_size 128 --model resnet50 --feat_dim 128 --temp 0.1
```

Quick smoke test (1 epoch, 5 steps):

```bash
python src/train_rascal.py --method rascal --dataset_name cifar10 --data_root ./data \
	--epochs 1 --max_steps 5 --batch_size 64
```

Checkpoints are written under `./save/rascal/.../`.

## 2) Linear probe

After pretraining, point `--ckpt` to `ckpt_last.pth` (or any saved epoch).

Some datasets may not have a `val` split; for CIFAR, use `--val_split test`.

```bash
python src/train_linear.py --ckpt ./save/rascal/<RUN_NAME>/ckpt_last.pth \
	--dataset_name cifar10 --data_root ./data --train_split train --val_split test \
	--epochs 50 --batch_size 128 --model resnet50
```

Output includes **Top-1** and **Top-5** accuracy.

Note: linear probe runs on CPU or GPU. CPU is supported but can be very slow.