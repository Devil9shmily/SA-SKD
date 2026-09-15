import pytest
import torch

from uav_kd.data import GraphDataset, resampled_binary_subset


def test_graph_dataset_and_resampling(tmp_path):
    path = tmp_path / "train.pt"
    torch.save(
        {
            "x": torch.randn(8, 20, 40, 11),
            "y": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1]),
        },
        path,
    )
    dataset = GraphDataset(path)
    subset = resampled_binary_subset(dataset, positive_repeat=2, negative_to_positive=1.0)
    labels = torch.tensor([int(subset[index][1]) for index in range(len(subset))])
    assert len(dataset) == 8
    assert int((labels == 1).sum()) == 4
    assert int((labels == 0).sum()) == 4


def test_graph_dataset_rejects_wrong_feature_count(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"x": torch.randn(2, 20, 40, 10), "y": torch.tensor([0, 1])}, path)
    with pytest.raises(ValueError, match="11 input features"):
        GraphDataset(path)
