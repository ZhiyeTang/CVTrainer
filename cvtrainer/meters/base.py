from abc import ABC, abstractmethod
from typing import Any


class BaseMeter(ABC):
    """Meter 基类"""

    @abstractmethod
    def update(self, value: Any):
        pass

    @abstractmethod
    def get_value(self) -> Any:
        pass

    @abstractmethod
    def reset(self):
        pass
