import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module):
    """三段式模型：backbone + adapter + head"""

    def __init__(
        self, backbone: "BaseBackbone", adapter: "BaseAdapter", head: "BaseHead"
    ):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.adapter(x)
        x = self.head(x)
        return x


class BaseBackbone(nn.Module, ABC):
    """Backbone 基类"""

    @property
    @abstractmethod
    def backbone_channel(self) -> int:
        """Backbone 输出通道数"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BaseAdapter(nn.Module, ABC):
    """Adapter 基类"""

    @property
    @abstractmethod
    def head_channel(self) -> int:
        """Adapter 输出通道数"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BaseHead(nn.Module, ABC):
    """Head 基类"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
