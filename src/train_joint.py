from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from ufgvc import UFGVCDataset
from resnet_big import SupConResNet, LinearClassifier
from supcon_loss import SupConLoss
from train_utils import (
    TwoCropTransform,
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    warmup_learning_rate,
)


def parse_option():
    parser = argparse.ArgumentParser("SupCon + CE Joint Training (UFGVC)")

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

    # dataset
    parser.add_argument("--dataset_name", type=str, default="soybean")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--download", type=int, default=1)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--mean", type=str, default="(0.485, 0.456, 0.406)")
    parser.add_argument("--std", type=str, default="(0.229, 0.224, 0.225)")

    # model
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--head", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--feat_dim", type=int, default=128)

    # losses
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--lambda_supcon", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=1.0)

    # other
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--trial", type=str, default="0")
    parser.add_argument("--max_steps", type=int, default=-1)

    opt = parser.parse_args()

    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(",") if x.strip()]
    opt.mean = eval(opt.mean)
    opt.std = eval(opt.std)

    opt.model_name = (
        f"joint_{opt.dataset_name}_{opt.model}_lr_{opt.learning_rate}_decay_{opt.weight_decay}"
        f"_bsz_{opt.batch_size}_temp_{opt.temp}_lamS_{opt.lambda_supcon}_lamCE_{opt.lambda_ce}"
        f"_trial_{opt.trial}"
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

    opt.save_folder = os.path.join("./save/joint", opt.model_name)
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
        return_index=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, len(train_dataset.classes)


def set_model_and_losses(opt, num_classes: int):
    model = SupConResNet(name=opt.model, head=opt.head, feat_dim=opt.feat_dim)
    classifier = LinearClassifier(name=opt.model, num_classes=num_classes, in_dim=int(model.encoder.num_features))

    supcon_criterion = SupConLoss(temperature=opt.temp, base_temperature=opt.temp)
    ce_criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        classifier = classifier.cuda()
        supcon_criterion = supcon_criterion.cuda()
        ce_criterion = ce_criterion.cuda()
        cudnn.benchmark = True
    else:
        raise NotImplementedError("This script currently requires CUDA")

    return model, classifier, supcon_criterion, ce_criterion


def save_ckpt(model, classifier, optimizer, opt, epoch, save_file):
    state = {
        "opt": vars(opt),
        "model": model.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_file)


def train_one_epoch(train_loader, model, classifier, supcon_criterion, ce_criterion, optimizer, epoch, opt):
    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_supcon_meter = AverageMeter()
    loss_ce_meter = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for step, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.max_steps > 0 and step >= opt.max_steps:
            break

        images = torch.cat([images[0], images[1]], dim=0)
        labels = labels.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, step, len(train_loader), optimizer)

        # encoder feats for CE
        feats = model.encoder(images)
        logits = classifier(feats)
        logit1, logit2 = torch.split(logits, [bsz, bsz], dim=0)
        logits_avg = 0.5 * (logit1 + logit2)
        loss_ce = ce_criterion(logits_avg, labels)

        # projection feats for SupCon
        proj = F.normalize(model.head(feats), dim=1)
        p1, p2 = torch.split(proj, [bsz, bsz], dim=0)
        proj_views = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
        loss_supcon = supcon_criterion(proj_views, labels)

        loss = opt.lambda_ce * loss_ce + opt.lambda_supcon * loss_supcon

        loss_meter.update(loss.item(), bsz)
        loss_supcon_meter.update(loss_supcon.item(), bsz)
        loss_ce_meter.update(loss_ce.item(), bsz)

        acc1, _ = accuracy(logits_avg, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

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
                f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t"
                f"supcon {loss_supcon_meter.val:.3f} ({loss_supcon_meter.avg:.3f})\t"
                f"ce {loss_ce_meter.val:.3f} ({loss_ce_meter.avg:.3f})\t"
                f"Acc@1 {top1.val:.2f} ({top1.avg:.2f})"
            )
            sys.stdout.flush()

    return loss_meter.avg


def main():
    opt = parse_option()
    train_loader, n_cls = set_loader(opt)
    model, classifier, supcon_criterion, ce_criterion = set_model_and_losses(opt, num_classes=n_cls)

    optimizer = optim.SGD(
        list(model.parameters()) + list(classifier.parameters()),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        loss = train_one_epoch(train_loader, model, classifier, supcon_criterion, ce_criterion, optimizer, epoch, opt)
        print(f"epoch {epoch}, loss {loss:.4f}")

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{epoch}.pth")
            save_ckpt(model, classifier, optimizer, opt, epoch, save_file)

    save_file = os.path.join(opt.save_folder, "ckpt_last.pth")
    save_ckpt(model, classifier, optimizer, opt, opt.epochs, save_file)
    print(f"Saved: {save_file}")


if __name__ == "__main__":
    main()
