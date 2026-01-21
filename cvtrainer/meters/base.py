from abc import ABC, abstractmethod
from typing import Any, Union, Dict
from numbers import Number


class BaseMeter(ABC):
    """Meter 基类"""

    @abstractmethod
    def update(self, value: Any):
        pass

    @abstractmethod
    def get_value(self) -> Union[Number, Dict[str, Number]]:
        """
        获取指标值

        Returns:
            可以返回单个数值（如 LossMeter）或字典（如 DetectionMapMeter）
            返回字典时，key 应该能清晰表示指标含义
        """
        pass

    @abstractmethod
    def reset(self):
        pass
