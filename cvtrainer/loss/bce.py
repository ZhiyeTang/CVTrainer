import torch.nn as nn


class BCELoss(nn.Module):
    """BCE Loss"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.loss_fn(output, target)
