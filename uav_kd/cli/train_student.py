"""Train the student without knowledge distillation."""

from __future__ import annotations

import argparse

from uav_kd.models import StudentSTGCN

from .common import add_training_arguments
from .supervised import run_supervised


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(
        parser,
        output="outputs/checkpoints/student_baseline.pt",
        learning_rate=1e-4,
    )
    return parser


def main() -> None:
    run_supervised(
        "Student baseline training",
        build_parser().parse_args(),
        StudentSTGCN(),
        cosine_schedule=False,
    )


if __name__ == "__main__":
    main()
