"""Small training utilities (meters, LR schedule, transforms wrapper)."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python/NumPy/PyTorch for reproducibility.

    Notes:
        - Pass a negative seed (e.g. -1) to disable seeding.
        - If deterministic=True, PyTorch may raise on non-deterministic ops.
    """

    if seed is None or seed < 0:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Must be set before the first CUDA context is created to be effective.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function to make random transforms reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TwoCropTransform:
    """Create two crops/views of the same image."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(opt, optimizer, epoch):
    lr = opt.learning_rate
    if opt.cosine:
        eta_min = lr * (opt.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / opt.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        if steps > 0:
            lr = lr * (opt.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(opt, epoch, batch_id, total_batches, optimizer):
    if opt.warm and epoch <= opt.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (opt.warm_epochs * total_batches)
        lr = opt.warmup_from + p * (opt.warmup_to - opt.warmup_from)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def set_optimizer(opt, model):
    return optim.SGD(
        model.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )


def save_model(model, optimizer, opt, epoch, save_file):
    save_file = str(save_file)
    state = {
        "opt": vars(opt),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    Path(save_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_file)
