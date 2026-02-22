from __future__ import annotations

import argparse
import math
import os
import sys
import time

# Allow running as: `python src/train_rascal.py` from the repo root.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from ufgvc import UFGVCDataset
from resnet_big import SupConResNet
from supcon_loss import SupConLoss
from rascal_loss import RASCALLoss
from train_utils import (
    TwoCropTransform,
    AverageMeter,
    adjust_learning_rate,
    warmup_learning_rate,
    set_optimizer,
    save_model,
    set_seed,
    seed_worker,
)


def parse_option():
    parser = argparse.ArgumentParser("RASCAL pretraining")

    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    # dataset (UFGVC)
    parser.add_argument("--dataset_name", type=str, default="soybean")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--download",
        type=int,
        default=1,
        help="1=download dataset if missing, 0=do not download",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument(
        "--mean", type=str, default="(0.485, 0.456, 0.406)", help="normalize mean"
    )
    parser.add_argument(
        "--std", type=str, default="(0.229, 0.224, 0.225)", help="normalize std"
    )

    # model
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--head", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--feat_dim", type=int, default=128)

    # method
    parser.add_argument(
        "--method",
        type=str,
        default="rascal",
        choices=["rascal", "supcon"],
        help="pretrain objective",
    )
    parser.add_argument("--temp", type=float, default=0.1)

    # other
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--trial", type=str, default="0")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed; set to -1 to disable seeding",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="force deterministic algorithms (slower; may error on nondeterministic ops)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="stop after N steps per epoch (for quick smoke tests)",
    )

    opt = parser.parse_args()

    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(",") if x.strip()]
    opt.mean = eval(opt.mean)
    opt.std = eval(opt.std)

    opt.model_name = (
        f"{opt.method}_{opt.dataset_name}_{opt.model}_lr_{opt.learning_rate}"
        f"_decay_{opt.weight_decay}_bsz_{opt.batch_size}_temp_{opt.temp}_trial_{opt.trial}"
    )
    if opt.cosine:
        opt.model_name = f"{opt.model_name}_cosine"

    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = f"{opt.model_name}_warm"
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)
            ) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join("./save/rascal", opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def set_loader(opt):
    normalize = transforms.Normalize(mean=opt.mean, std=opt.std)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = UFGVCDataset(
        dataset_name=opt.dataset_name,
        root=opt.data_root,
        split=opt.split,
        transform=TwoCropTransform(train_transform),
        download=bool(opt.download),
        return_index=True,
    )

    generator = None
    if opt.seed is not None and opt.seed >= 0:
        generator = torch.Generator()
        generator.manual_seed(opt.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker if generator is not None else None,
        generator=generator,
    )

    return train_loader


def set_model_and_loss(opt, num_samples: int):
    model = SupConResNet(name=opt.model, head=opt.head, feat_dim=opt.feat_dim)
    if opt.method == "supcon":
        criterion = SupConLoss(temperature=opt.temp, base_temperature=opt.temp)
    else:
        criterion = RASCALLoss(
            temperature=opt.temp,
            base_temperature=opt.temp,
            num_samples=num_samples,
            feat_dim=opt.feat_dim,
            persistent_cache=False,
        )

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = not opt.deterministic
        cudnn.deterministic = bool(opt.deterministic)

    return model, criterion


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.max_steps > 0 and step >= opt.max_steps:
            break

        if len(batch) != 3:
            raise ValueError("Expected (images, labels, indices) from dataset")

        images, labels, indices = batch
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            indices = indices.cuda(non_blocking=True)

        bsz = labels.shape[0]
        warmup_learning_rate(opt, epoch, step, len(train_loader), optimizer)

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if opt.method == "supcon":
            loss = criterion(features, labels)
        else:
            loss = criterion(features, labels, indices)

        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % opt.print_freq == 0:
            print(
                f"Train: [{epoch}][{step+1}/{len(train_loader)}]\t"
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"loss {losses.val:.3f} ({losses.avg:.3f})"
            )
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    set_seed(opt.seed, deterministic=opt.deterministic)
    if opt.seed is not None and opt.seed >= 0:
        print(f"Seed: {opt.seed} (deterministic={opt.deterministic})")

    train_loader = set_loader(opt)
    model, criterion = set_model_and_loss(opt, num_samples=len(train_loader.dataset))
    optimizer = set_optimizer(opt, model)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, opt)
        print(f"epoch {epoch}, loss {loss:.4f}")

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{epoch}.pth")
            save_model(model, optimizer, opt, epoch, save_file)

    # always save last
    save_file = os.path.join(opt.save_folder, "ckpt_last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)
    print(f"Saved: {save_file}")


if __name__ == "__main__":
    main()
