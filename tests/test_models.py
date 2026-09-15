import pytest
import torch

from uav_kd.distillation import (
    FeatureAdapters,
    attention_distillation_loss,
    feature_distillation_loss,
)
from uav_kd.models import StudentSTGCN, TeacherSTGCN


@pytest.fixture(scope="module")
def model_outputs():
    torch.manual_seed(0)
    x = torch.randn(2, 11, 4, 5)
    student = StudentSTGCN().eval()
    teacher = TeacherSTGCN().eval()
    with torch.no_grad():
        student_logits, student_features = student(x, return_features=True)
        teacher_logits, teacher_features = teacher(x, return_features=True)
    return student, teacher, student_logits, teacher_logits, student_features, teacher_features


def test_parameter_counts_match_paper_scale(model_outputs):
    student, teacher, *_ = model_outputs
    student_count = sum(parameter.numel() for parameter in student.parameters())
    teacher_count = sum(parameter.numel() for parameter in teacher.parameters())
    assert 3_000_000 < student_count < 3_200_000
    assert 12_300_000 < teacher_count < 12_500_000


def test_feature_and_attention_interfaces(model_outputs):
    _, _, student_logits, teacher_logits, student_features, teacher_features = model_outputs
    assert student_logits.shape == teacher_logits.shape == (2,)
    assert student_features["block1"].shape == (2, 64, 4, 5)
    assert teacher_features["block1"].shape == (2, 128, 4, 5)
    for name in ("ta1", "sa2", "ta3", "sa3"):
        assert student_features[name].shape == teacher_features[name].shape == (2, 1, 4, 5)


def test_paper_distillation_losses_are_finite(model_outputs):
    _, _, _, _, student_features, teacher_features = model_outputs
    adapters = FeatureAdapters()
    feature_loss = feature_distillation_loss(
        student_features, teacher_features, adapters, mode="b1_b3"
    )
    attention_loss = attention_distillation_loss(
        student_features, teacher_features, mode="ta_b1_sa_b2"
    )
    assert torch.isfinite(feature_loss)
    assert torch.isfinite(attention_loss)
