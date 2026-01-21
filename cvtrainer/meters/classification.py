import torch
from typing import Dict, Union
from numbers import Number
from .base import BaseMeter


class ClassificationMetricsMeter(BaseMeter):
    """
    分类指标 Meter - F1, Precision, Recall

    Args:
        num_classes: 类别数量
        average: 'macro' 或 'micro'
    """

    def __init__(self, num_classes: int, average: str = "macro"):
        self.num_classes = num_classes
        self.average = average

        if average not in ["macro", "micro"]:
            raise ValueError(f"average 必须是 'macro' 或 'micro', 实际: {average}")

        self.confusion_matrix = torch.zeros(num_classes, num_classes)

    def update(self, output: torch.Tensor, target: torch.Tensor):
        """
        更新混淆矩阵

        Args:
            output: (B, num_classes) - logits
            target: (B,) - 类别索引
        """
        pred = output.argmax(dim=1)

        for t, p in zip(target, pred):
            self.confusion_matrix[t, p] += 1

    def _compute_metrics(self) -> Dict[str, float]:
        """计算分类指标"""
        if self.average == "micro":
            tp = self.confusion_matrix.diag().sum().item()
            fp = self.confusion_matrix.sum(dim=0).sum().item() - tp
            fn = self.confusion_matrix.sum(dim=1).sum().item() - tp

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1)
        else:
            precisions = []
            recalls = []
            f1s = []

            for i in range(self.num_classes):
                tp = self.confusion_matrix[i, i].item()
                fp = self.confusion_matrix[:, i].sum().item() - tp
                fn = self.confusion_matrix[i, :].sum().item() - tp

                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            precision = sum(precisions) / len(precisions)
            recall = sum(recalls) / len(recalls)
            f1 = sum(f1s) / len(f1s)

        return {"precision": precision * 100, "recall": recall * 100, "f1": f1 * 100}

    def get_value(self) -> Dict[str, float]:
        return self._compute_metrics()

    def reset(self):
        self.confusion_matrix.zero_()


class F1Meter(ClassificationMetricsMeter):
    """F1 Score Meter"""

    def get_value(self) -> float:
        metrics = self._compute_metrics()
        return metrics["f1"]


class PrecisionMeter(ClassificationMetricsMeter):
    """Precision Meter"""

    def get_value(self) -> float:
        metrics = self._compute_metrics()
        return metrics["precision"]


class RecallMeter(ClassificationMetricsMeter):
    """Recall Meter"""

    def get_value(self) -> float:
        metrics = self._compute_metrics()
        return metrics["recall"]
