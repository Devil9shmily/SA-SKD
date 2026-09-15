# Dataset format

Create one directory for the source domain and one for the UAV target domain:

```text
data/
├── source/
│   ├── train.pt
│   ├── val.pt
│   └── test.pt
└── target/
    ├── train.pt
    ├── val.pt
    └── test.pt
```

Each file must be a PyTorch mapping with two tensors:

- `x`: shape `[samples, 20, 40, 11]` (`time, nodes, features`)
- `y`: shape `[samples]`, with binary labels `0` (normal) and `1` (accident)

The graph builder reads position from feature indices `0:2` and velocity from
indices `6:8`. The remaining features must follow the same order and
normalization used to create the paper's training data.

Dataset files are intentionally ignored by Git. Only load `.pt` files that you
created yourself or obtained from a trusted source because older PyTorch
versions may use pickle while loading them.
