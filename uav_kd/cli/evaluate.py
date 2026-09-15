"""Evaluate a teacher or student checkpoint on one dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_kd.data import GraphDataset, make_loader
from uav_kd.metrics import evaluate
from uav_kd.models import StudentSTGCN, TeacherSTGCN
from uav_kd.utils import load_model_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("student", "teacher"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Path to a .pt split.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = (StudentSTGCN() if args.model == "student" else TeacherSTGCN()).to(device)
    keys = ("student", "model") if args.model == "student" else ("teacher", "model")
    load_model_checkpoint(model, args.checkpoint, device, keys=keys)
    dataset = GraphDataset(args.data)
    print(dataset.summary())
    loader = make_loader(
        dataset,
        args.batch_size,
        shuffle=False,
        workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    print(evaluate(model, loader, device))


if __name__ == "__main__":
    main()
