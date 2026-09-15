"""Stage 2: target-domain selective attention distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uav_kd.data import resampled_binary_subset
from uav_kd.distillation import ATTENTION_MODES, attention_distillation_loss
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
        output="outputs/checkpoints/stage2_ta_b1_sa_b2.pt",
        learning_rate=1e-5,
    )
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(ATTENTION_MODES), default="ta_b1_sa_b2")
    parser.add_argument(
        "--attention-weight",
        type=float,
        default=0.5,
        help="Global multiplier for the sum of selected attention losses.",
    )
    parser.add_argument(
        "--resample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the class resampling used in the original draft script.",
    )
    parser.add_argument("--positive-repeat", type=int, default=2)
    parser.add_argument("--negative-to-positive", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    def maybe_resample(dataset):
        if not args.resample:
            return dataset
        return resampled_binary_subset(
            dataset,
            positive_repeat=args.positive_repeat,
            negative_to_positive=args.negative_to_positive,
            seed=args.seed,
        )

    train_loader, val_loader, test_loader = build_loaders(
        args, device, train_dataset=maybe_resample if args.resample else None
    )
    teacher = TeacherSTGCN().to(device)
    student = StudentSTGCN().to(device)
    load_model_checkpoint(teacher, args.teacher_checkpoint, device, keys=("model", "teacher"))
    load_model_checkpoint(student, args.student_checkpoint, device, keys=("student", "model"))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    criterion = binary_criterion(args.pos_weight, device)
    optimizer = torch.optim.Adam(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    print_run_header(
        "Stage 2 - selective attention distillation",
        args,
        device,
        {"Frozen teacher": teacher, "Student": student},
    )

    best_f1 = float("-inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = classification_sum = attention_sum = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).permute(0, 3, 1, 2)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, teacher_features = teacher(x, return_features=True)
            student_logits, student_features = student(x, return_features=True)
            classification_loss = criterion(student_logits, y)
            attention_loss = attention_distillation_loss(
                student_features, teacher_features, args.mode
            )
            loss = classification_loss + args.attention_weight * attention_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            classification_sum += classification_loss.item()
            attention_sum += attention_loss.item()

        batches = max(len(train_loader), 1)
        validation = evaluate(student, val_loader, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | loss {total_loss / batches:.4f} "
            f"(cls {classification_sum / batches:.4f}, att {attention_sum / batches:.4f}) "
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
