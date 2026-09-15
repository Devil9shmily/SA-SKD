# Two-Stage Selective Cross-Domain Distillation for UAV Accident Detection

Research-code structure for **Two-Stage Selective Cross-Domain Distillation for
Lightweight UAV Traffic Accident Detection**.

The method separates source-domain knowledge inheritance from target-domain
adaptation:

1. Train a 12.4M-parameter teacher on source-domain traffic accidents.
2. Distill Block 1 and Block 3 features into a 3.1M-parameter student.
3. Adapt that student to the UAV domain using Block 1 temporal attention and
   Block 2 spatial attention.

The paper reports an F1-score of **0.9209** and AUC of **0.9786** for the final
student. Results are not bundled in this repository; they require the same
processed data, splits, and training environment.

## Repository layout

```text
. 
├── data/                     # Expected data format (tensor files are ignored)
├── tests/                    # Model, loss, and dataset smoke tests
├── uav_kd/
│   ├── cli/                  # Training and evaluation entry points
│   ├── models/               # Dynamic graph, teacher, and student
│   ├── data.py
│   ├── distillation.py
│   ├── metrics.py
│   ├── training.py
│   └── utils.py
├── CITATION.cff
├── pyproject.toml
└── requirements.txt
```

## Installation

Python 3.10 or newer is required. Create an isolated environment, install the
PyTorch build appropriate for your CPU/CUDA setup, and install this project:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Data

Prepare source- and target-domain splits as described in
[data/README.md](data/README.md):

```text
data/source/{train,val,test}.pt
data/target/{train,val,test}.pt
```

Each `.pt` file is a mapping containing `x` with shape
`[samples, 20, 40, 11]` and binary `y` with shape `[samples]`. Dataset files,
checkpoints, and outputs are excluded by `.gitignore`.

The original preprocessing pipeline is not included in the supplied code. To
reproduce the paper exactly, use the same feature order, normalization, and
train/validation/test splits used in the experiments.

## Training

Run all commands from the repository root after installation.

### 1. Source-domain teacher

```bash
uav-kd-train-teacher --data-root data/source
```

The default checkpoint is `outputs/checkpoints/teacher.pt`.

### 2. Stage 1: selective feature distillation

```bash
uav-kd-stage1 \
  --data-root data/source \
  --teacher-checkpoint outputs/checkpoints/teacher.pt \
  --mode b1_b3
```

The default `b1_b3` mode implements the paper's complementary low-level and
high-level feature transfer. Available ablations are `b1`, `b2`, `b3`,
`b1_b2`, `b1_b3`, `b2_b3`, and `b1_b2_b3`.

### 3. Stage 2: UAV-domain attention distillation

```bash
uav-kd-stage2 \
  --data-root data/target \
  --teacher-checkpoint outputs/checkpoints/teacher.pt \
  --student-checkpoint outputs/checkpoints/stage1_b1_b3.pt \
  --mode ta_b1_sa_b2
```

The default mode is the paper's best selection. Other modes reproduce the
attention ablations: `base`, `ta_b1`, `sa_b2`, `ta_b3`, `ta_b1_ta_b3`, and
`sa_b2_ta_b3`.

The teacher is frozen in both distillation stages. The draft's optional target
class resampling can be enabled with `--resample`; it is disabled by default
because it is not specified in the paper's implementation details.

### Baseline and evaluation

```bash
uav-kd-train-student --data-root data/target
uav-kd-evaluate --model student --checkpoint outputs/checkpoints/stage2_ta_b1_sa_b2.pt --data data/target/test.pt
```

All training options are configurable from the CLI. Run any command with
`--help` to see learning rates, loss weights, early stopping, device selection,
and output paths.

## Verification

```bash
pytest
ruff check .
```

The tests check the paper-scale parameter counts, model outputs, intermediate
feature/attention interfaces, selective losses, and dataset validation.

## Implementation note

The supplied draft referenced a missing `student_model.py` while Stage 1 used a
different depthwise Student-C variant. The packaged student restores the
paper-scale architecture by using residual temporal convolution units at
64/128/256 channels; this yields approximately 3.1M parameters and makes the
two stages use one compatible student. Legacy draft files are kept locally in
the ignored `legacy/` directory and are not part of the public package.
```
