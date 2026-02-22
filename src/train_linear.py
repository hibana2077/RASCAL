from __future__ import annotations

import argparse
import math
import os
import sys
import time

# Allow running as: `python src/train_linear.py` from the repo root.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from ufgvc import UFGVCDataset
from resnet_big import SupConResNet, LinearClassifier
from train_utils import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    warmup_learning_rate,
    set_optimizer,
)


def parse_option():
    parser = argparse.ArgumentParser("Linear evaluation (UFGVC)")

    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--learning_rate", type=float, default=5.0)
    parser.add_argument("--lr_decay_epochs", type=str, default="30,40")
    parser.add_argument("--lr_decay_rate", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--head", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--dataset_name", type=str, default="soybean")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument(
        "--download",
        type=int,
        default=1,
        help="1=download dataset if missing, 0=do not download",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--mean", type=str, default="(0.485, 0.456, 0.406)")
    parser.add_argument("--std", type=str, default="(0.229, 0.224, 0.225)")

    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--warm", action="store_true")

    opt = parser.parse_args()
    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(",") if x.strip()]
    opt.mean = eval(opt.mean)
    opt.std = eval(opt.std)

    # warm-up (match the pretrain scripts' behavior)
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)
            ) / 2
        else:
            opt.warmup_to = opt.learning_rate

    return opt


def set_loader(opt):
    normalize = transforms.Normalize(mean=opt.mean, std=opt.std)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((opt.size, opt.size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = UFGVCDataset(
        dataset_name=opt.dataset_name,
        root=opt.data_root,
        split=opt.train_split,
        transform=train_transform,
        download=bool(opt.download),
        return_index=False,
    )

    # Some datasets might not have 'val'; if so, user can set --val_split test.
    val_dataset = UFGVCDataset(
        dataset_name=opt.dataset_name,
        root=opt.data_root,
        split=opt.val_split,
        transform=val_transform,
        download=bool(opt.download),
        return_index=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, len(train_dataset.classes)


def set_model(opt, num_classes: int):
    model = SupConResNet(name=opt.model, head=opt.head, feat_dim=opt.feat_dim)
    classifier = LinearClassifier(name=opt.model, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ckpt = torch.load(opt.ckpt, map_location="cpu")
    state_dict = ckpt["model"]

    # Robustly handle checkpoints saved with DataParallel.
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("encoder.module.", "encoder.")
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict

    if device.type == "cuda":
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        cudnn.benchmark = True

    model = model.to(device)
    classifier = classifier.to(device)
    criterion = criterion.to(device)
    model.load_state_dict(state_dict, strict=True)

    return model, classifier, criterion, device


def train_one_epoch(train_loader, model, classifier, criterion, optimizer, epoch, opt, device):
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        losses.update(loss.item(), bsz)
        acc1, _ = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print(
                f"Train: [{epoch}][{idx+1}/{len(train_loader)}]\t"
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"loss {losses.val:.3f} ({losses.avg:.3f})\t"
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
            )
            sys.stdout.flush()

    return losses.avg, top1.avg


@torch.no_grad()
def validate(val_loader, model, classifier, criterion, opt, device):
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(val_loader):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))
        bsz = labels.shape[0]

        output = classifier(model.encoder(images))
        loss = criterion(output, labels)

        losses.update(loss.item(), bsz)
        acc1, _ = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print(
                f"Test: [{idx+1}/{len(val_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
            )

    print(f" * Acc@1 {top1.avg:.3f}")
    return losses.avg, top1.avg


def main():
    best_acc = 0.0
    opt = parse_option()

    train_loader, val_loader, n_cls = set_loader(opt)
    model, classifier, criterion, device = set_model(opt, num_classes=n_cls)
    optimizer = set_optimizer(opt, classifier)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        loss, acc = train_one_epoch(train_loader, model, classifier, criterion, optimizer, epoch, opt, device)
        print(f"Train epoch {epoch}, loss {loss:.4f}, acc@1 {float(acc):.2f}")

        val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt, device)
        best_acc = max(best_acc, float(val_acc))
        print(f"Val epoch {epoch}, loss {val_loss:.4f}, acc@1 {float(val_acc):.2f}")

    print(f"best accuracy: {best_acc:.2f}")


if __name__ == "__main__":
    main()
