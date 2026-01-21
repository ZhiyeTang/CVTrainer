import torch
from typing import Dict, Union
from numbers import Number
from .base import BaseMeter


class SegmentationIoUMeter(BaseMeter):
    """
    语义分割 IoU Meter

    Args:
        num_classes: 类别数量
        ignore_index: 忽略的类别索引（如255用于边界）
    """

    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.intersection = torch.zeros(num_classes)
        self.union = torch.zeros(num_classes)
        self.count = torch.zeros(num_classes)

    def update(self, output: torch.Tensor, target: torch.Tensor):
        """
        更新 IoU 统计

        Args:
            output: (B, C, H, W) - logits 或概率
            target: (B, H, W) - 类别索引
        """
        pred = output.argmax(dim=1)

        mask = target != self.ignore_index

        for cls in range(self.num_classes):
            pred_cls = (pred == cls) & mask
            target_cls = (target == cls) & mask

            intersection = (pred_cls & target_cls).sum().item()
            union = (pred_cls | target_cls).sum().item()

            self.intersection[cls] += intersection
            self.union[cls] += union
            self.count[cls] += (target_cls).sum().item()

    def get_value(self) -> float:
        """
        计算 mean IoU

        Returns:
            mIoU: 所有有效类别的IoU平均值
        """
        ious = []

        for cls in range(self.num_classes):
            if self.count[cls] > 0:
                iou = self.intersection[cls] / max(self.union[cls], 1)
                ious.append(iou)

        if len(ious) == 0:
            return 0.0

        return sum(ious) / len(ious) * 100

    def reset(self):
        self.intersection.zero_()
        self.union.zero_()
        self.count.zero_()
