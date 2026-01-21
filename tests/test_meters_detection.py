import torch
import pytest
from cvtrainer.meters import DetectionMapMeter


def test_map_meter_perfect_predictions():
    """测试 mAP Meter 在完美预测情况下的表现"""
    meter = DetectionMapMeter(num_classes=2)

    preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
            "scores": torch.tensor([0.9]),
        }
    ]
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}]
    meter.update(preds, targets)

    result = meter.get_value()
    assert result["map_50"] == pytest.approx(100.0, rel=1e-5)


def test_map_meter_no_predictions():
    """测试 mAP Meter 在没有预测情况下的表现"""
    meter = DetectionMapMeter(num_classes=2)

    preds = [
        {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "scores": torch.zeros(0),
        }
    ]
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}]
    meter.update(preds, targets)

    result = meter.get_value()
    assert result["map"] == 0.0
    assert result["map_50"] == 0.0


def test_map_meter_reset():
    """测试 mAP Meter 的重置功能"""
    meter = DetectionMapMeter(num_classes=2)

    preds1 = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
            "scores": torch.tensor([0.9]),
        }
    ]
    targets1 = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}]
    meter.update(preds1, targets1)

    meter.reset()

    preds2 = [
        {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "scores": torch.zeros(0),
        }
    ]
    targets2 = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}]
    meter.update(preds2, targets2)

    result = meter.get_value()
    assert result["map_50"] == 0.0


def test_map_meter_empty():
    """测试 mAP Meter 在空数据情况下的表现"""
    meter = DetectionMapMeter(num_classes=2)

    result = meter.get_value()
    assert result["map"] == 0.0
    assert result["map_50"] == 0.0
    assert result["map_75"] == 0.0


def test_map_meter_multiple_classes():
    """测试 mAP Meter 在多类别情况下的表现"""
    meter = DetectionMapMeter(num_classes=3)

    preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
            "scores": torch.tensor([0.9]),
        },
        {
            "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.8]),
        },
    ]
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]), "labels": torch.tensor([1])},
    ]
    meter.update(preds, targets)

    result = meter.get_value()
    assert result["map_50"] > 0.0


def test_map_meter_wrong_predictions():
    """测试 mAP Meter 在预测错误情况下的表现"""
    meter = DetectionMapMeter(num_classes=2)

    preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "labels": torch.tensor([0]),
            "scores": torch.tensor([0.9]),
        }
    ]
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}]
    meter.update(preds, targets)

    result = meter.get_value()
    assert result["map_50"] < 100.0
