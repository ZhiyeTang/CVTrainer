import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAdapter


class LinearAdapter(BaseAdapter):
    """Linear Adapter: Global Average Pooling + Linear"""

    def __init__(self, backbone_channel: int, head_channel: int):
        super().__init__()
        self._backbone_channel = backbone_channel
        self._head_channel = head_channel
        self.fc = nn.Linear(backbone_channel, head_channel)

    @property
    def head_channel(self) -> int:
        return self._head_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
