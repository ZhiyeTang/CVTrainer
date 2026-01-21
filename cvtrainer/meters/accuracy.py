import torch
from typing import Dict, Union
from numbers import Number
from .base import BaseMeter


class AccuracyMeter(BaseMeter):
    """准确率 Meter"""

    def __init__(self, topk: tuple = (1,)):
        self.topk = topk
        self.correct: Dict[int, int] = {k: 0 for k in topk}
        self.total = 0

    def update(self, output: torch.Tensor, target: torch.Tensor):
        """更新准确率"""
        maxk = max(self.topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        self.total += target.size(0)
        for k in self.topk:
            self.correct[k] += correct[:k].reshape(-1).float().sum(0, keepdim=True).item()

    def get_value(self) -> Dict[str, Number]:
        """获取准确率"""
        return {f"accuracy_{k}": 100.0 * self.correct[k] / max(self.total, 1) for k in self.topk}

    def reset(self):
        """重置"""
        self.correct = {k: 0 for k in self.topk}
        self.total = 0
