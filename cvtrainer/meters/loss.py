from .base import BaseMeter


class LossMeter(BaseMeter):
    """Loss Meter"""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float):
        """更新 loss"""
        self.sum += value
        self.count += 1

    def get_value(self) -> float:
        """获取平均 loss"""
        return self.sum / max(self.count, 1)

    def reset(self):
        """重置"""
        self.sum = 0.0
        self.count = 0
