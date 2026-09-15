"""Train the source-domain teacher."""

from __future__ import annotations

import argparse

from uav_kd.models import TeacherSTGCN

from .common import add_training_arguments
from .supervised import run_supervised


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(
        parser,
        output="outputs/checkpoints/teacher.pt",
        learning_rate=3e-5,
    )
    return parser


def main() -> None:
    run_supervised(
        "Source-domain teacher training",
        build_parser().parse_args(),
        TeacherSTGCN(),
        cosine_schedule=True,
    )


if __name__ == "__main__":
    main()
