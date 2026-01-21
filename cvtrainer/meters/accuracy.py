import torch
from .base import BaseMeter
from typing import List


class AccuracyMeter(BaseMeter):
    """准确率 Meter"""

    def __init__(self, topk: tuple = (1,)):
        self.topk = topk
        self.correct: List[int] = [0] * len(topk)
        self.total = 0

    def update(self, output: torch.Tensor, target: torch.Tensor):
        """更新准确率"""
        maxk = max(self.topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        self.total += target.size(0)
        for i, k in enumerate(self.topk):
            self.correct[i] += correct[:k].reshape(-1).float().sum(0, keepdim=True).item()

    def get_value(self) -> List[float]:
        """获取准确率"""
        return [100.0 * c / max(self.total, 1) for c in self.correct]

    def reset(self):
        """重置"""
        self.correct = [0] * len(self.topk)
        self.total = 0
