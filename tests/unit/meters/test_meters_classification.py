import torch
import pytest
from cvtrainer.meters import F1Meter, PrecisionMeter, RecallMeter


def test_f1_meter_perfect_predictions():
    """测试 F1 Meter 在完美预测情况下的表现"""
    meter = F1Meter(num_classes=3, average="macro")

    output = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    target = torch.tensor([0, 1, 2])
    meter.update(output, target)

    assert meter.get_value() == pytest.approx(100.0, rel=1e-5)


def test_f1_meter_all_wrong_predictions():
    """测试 F1 Meter 在全部预测错误情况下的表现"""
    meter = F1Meter(num_classes=2, average="macro")

    output = torch.tensor([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    target = torch.tensor([0, 0, 0, 0])
    meter.update(output, target)

    assert meter.get_value() == pytest.approx(0.0, abs=1e-5)


def test_f1_meter_reset():
    """测试 F1 Meter 的重置功能"""
    meter = F1Meter(num_classes=3, average="macro")

    output = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    target = torch.tensor([0, 1])
    meter.update(output, target)

    meter.reset()

    output2 = torch.tensor([[10.0, 0.0, 0.0]])
    target2 = torch.tensor([1])
    meter.update(output2, target2)

    assert meter.get_value() == pytest.approx(0.0, abs=1e-5)


def test_precision_meter():
    """测试 Precision Meter"""
    meter = PrecisionMeter(num_classes=3, average="macro")

    output = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    target = torch.tensor([0, 0, 1])
    meter.update(output, target)

    result = meter.get_value()
    assert result == pytest.approx(66.6667, rel=1e-2)


def test_recall_meter():
    """测试 Recall Meter"""
    meter = RecallMeter(num_classes=3, average="macro")

    output = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    target = torch.tensor([0, 0, 0])
    meter.update(output, target)

    result = meter.get_value()
    assert result == pytest.approx(33.3333, rel=1e-2)


def test_micro_average():
    """测试 micro 平均方式"""
    meter = F1Meter(num_classes=3, average="micro")

    output = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
    target = torch.tensor([0, 1, 2])
    meter.update(output, target)

    result = meter.get_value()
    assert result == pytest.approx(66.6667, rel=1e-2)
