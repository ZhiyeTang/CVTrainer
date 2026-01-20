import torch
import torch.nn as nn
from ..base import BaseAdapter


class IdentityAdapter(BaseAdapter):
    """Identity Adapter，不改变通道数"""

    def __init__(self, backbone_channel: int):
        super().__init__()
        self._head_channel = backbone_channel

    @property
    def head_channel(self) -> int:
        return self._head_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
