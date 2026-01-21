import torch
import pytest
from cvtrainer.meters import SegmentationIoUMeter


def test_miou_meter_perfect_predictions():
    """测试 IoU Meter 在完美预测情况下的表现"""
    meter = SegmentationIoUMeter(num_classes=3)

    output = torch.zeros(1, 3, 2, 2)
    output[0, 0, 0, 0] = 10.0
    output[0, 1, 0, 1] = 10.0
    output[0, 2, 1, 0] = 10.0
    output[0, 0, 1, 1] = 10.0

    target = torch.tensor([[[0, 1], [2, 0]]])
    meter.update(output, target)

    result = meter.get_value()
    assert result == pytest.approx(100.0, rel=1e-5)


def test_miou_meter_all_wrong_predictions():
    """测试 IoU Meter 在全部预测错误情况下的表现"""
    meter = SegmentationIoUMeter(num_classes=2)

    output = torch.zeros(1, 2, 2, 2)
    output[0, 0, :, :] = 10.0

    target = torch.tensor([[[1, 1], [1, 1]]])
    meter.update(output, target)

    result = meter.get_value()
    assert result == pytest.approx(0.0, abs=1e-5)


def test_miou_meter_reset():
    """测试 IoU Meter 的重置功能"""
    meter = SegmentationIoUMeter(num_classes=3)

    output1 = torch.zeros(1, 3, 2, 2)
    output1[0, 0, :, :] = 10.0
    target1 = torch.zeros(1, 2, 2, dtype=torch.long)
    meter.update(output1, target1)

    meter.reset()

    output2 = torch.zeros(1, 3, 2, 2)
    output2[0, 1, :, :] = 10.0
    target2 = torch.ones(1, 2, 2, dtype=torch.long)
    meter.update(output2, target2)

    result = meter.get_value()
    assert result == pytest.approx(100.0, rel=1e-5)


def test_miou_meter_with_ignore_index():
    """测试 IoU Meter 的 ignore_index 功能"""
    meter = SegmentationIoUMeter(num_classes=3, ignore_index=255)

    output = torch.zeros(1, 3, 3, 3)
    output[0, 0, :2, :2] = 10.0
    output[0, 1, :2, 2:] = 10.0
    output[0, 2, 2, :] = 10.0

    target = torch.tensor([[[0, 0, 255], [0, 0, 1], [255, 255, 2]]])
    meter.update(output, target)

    result = meter.get_value()
    assert result > 0.0


def test_miou_meter_multiple_batches():
    """测试 IoU Meter 在多个 batch 上的累积"""
    meter = SegmentationIoUMeter(num_classes=3)

    for _ in range(3):
        output = torch.zeros(1, 3, 2, 2)
        output[0, 0, :, :] = 10.0
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        meter.update(output, target)

    result = meter.get_value()
    assert result == pytest.approx(100.0, rel=1e-5)
