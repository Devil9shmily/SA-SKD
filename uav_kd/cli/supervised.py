"""Implementation shared by teacher and student baseline training."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from uav_kd.metrics import evaluate
from uav_kd.training import train_supervised_epoch
from uav_kd.utils import (
    load_model_checkpoint,
    print_run_header,
    resolve_device,
    save_checkpoint,
    serializable_config,
    seed_everything,
)

from .common import binary_criterion, build_loaders


def run_supervised(
    name: str,
    args: argparse.Namespace,
    model: nn.Module,
    cosine_schedule: bool,
) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model = model.to(device)
    train_loader, val_loader, test_loader = build_loaders(args, device)
    criterion = binary_criterion(args.pos_weight, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if cosine_schedule
        else None
    )
    print_run_header(name, args, device, {"Model": model})

    best_f1 = float("-inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_supervised_epoch(model, train_loader, optimizer, criterion, device)
        validation = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch:03d}/{args.epochs} | loss {loss:.4f} | val {validation}")

        if validation.f1 > best_f1:
            best_f1 = validation.f1
            stale_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_metrics": validation.as_dict(),
                    "config": serializable_config(args),
                },
                args.output,
            )
            print(f"Saved best checkpoint: {args.output}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    load_model_checkpoint(model, args.output, device, keys=("model",))
    print(f"Test | {evaluate(model, test_loader, device)}")
