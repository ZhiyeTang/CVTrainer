import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """交叉熵 Loss"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss_fn(output, target)
