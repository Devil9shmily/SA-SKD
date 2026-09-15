"""Stage 1: source-domain selective feature distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uav_kd.distillation import FEATURE_MODES, FeatureAdapters, feature_distillation_loss
from uav_kd.metrics import evaluate
from uav_kd.models import StudentSTGCN, TeacherSTGCN
from uav_kd.utils import (
    load_model_checkpoint,
    print_run_header,
    resolve_device,
    save_checkpoint,
    serializable_config,
    seed_everything,
)

from .common import add_training_arguments, binary_criterion, build_loaders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(
        parser,
        output="outputs/checkpoints/stage1_b1_b3.pt",
        learning_rate=1e-4,
    )
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(FEATURE_MODES), default="b1_b3")
    parser.add_argument(
        "--feature-weight",
        type=float,
        default=2.0,
        help="Global multiplier for the sum of selected feature losses.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_loader, val_loader, test_loader = build_loaders(args, device)

    teacher = TeacherSTGCN().to(device)
    student = StudentSTGCN().to(device)
    adapters = FeatureAdapters().to(device)
    load_model_checkpoint(teacher, args.teacher_checkpoint, device, keys=("model", "teacher"))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    criterion = binary_criterion(args.pos_weight, device)
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(adapters.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print_run_header(
        "Stage 1 - selective feature distillation",
        args,
        device,
        {"Teacher": teacher, "Student": student, "Adapters": adapters},
    )

    best_f1 = float("-inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        adapters.train()
        total_loss = classification_sum = feature_sum = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).permute(0, 3, 1, 2)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, teacher_features = teacher(x, return_features=True)
            student_logits, student_features = student(x, return_features=True)
            classification_loss = criterion(student_logits, y)
            feature_loss = feature_distillation_loss(
                student_features, teacher_features, adapters, args.mode
            )
            loss = classification_loss + args.feature_weight * feature_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            classification_sum += classification_loss.item()
            feature_sum += feature_loss.item()

        batches = max(len(train_loader), 1)
        validation = evaluate(student, val_loader, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | loss {total_loss / batches:.4f} "
            f"(cls {classification_sum / batches:.4f}, feat {feature_sum / batches:.4f}) "
            f"| val {validation}"
        )
        if validation.f1 > best_f1:
            best_f1 = validation.f1
            stale_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "mode": args.mode,
                    "student": student.state_dict(),
                    "adapters": adapters.state_dict(),
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

    load_model_checkpoint(student, args.output, device, keys=("student",))
    print(f"Test | {evaluate(student, test_loader, device)}")


if __name__ == "__main__":
    main()
