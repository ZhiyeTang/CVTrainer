import torch
import torch.nn as nn
from ..base import BaseAdapter


class ConvAdapter(BaseAdapter):
    """Conv Adapter: Conv1x1 + BN + ReLU"""

    def __init__(self, backbone_channel: int, head_channel: int):
        super().__init__()
        self._head_channel = head_channel
        self.conv = nn.Conv2d(backbone_channel, head_channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(head_channel)
        self.relu = nn.ReLU(inplace=True)

    @property
    def head_channel(self) -> int:
        return self._head_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
