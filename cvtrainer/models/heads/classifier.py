import torch
import torch.nn as nn
from ..base import BaseHead


class MulticlassClassifier(BaseHead):
    """多分类 Head"""

    def __init__(self, head_channel: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(head_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MultiLabelClassifier(BaseHead):
    """多标签分类 Head"""

    def __init__(self, head_channel: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(head_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        x = self.fc(x)
        return x
