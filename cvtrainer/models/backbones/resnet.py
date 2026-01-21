import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from ..base import BaseBackbone


class ResNet18Backbone(BaseBackbone):
    """ResNet18 Backbone"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.model = resnet18(weights=weights)
        self._backbone_channel = 512
        self.model.fc = nn.Identity()
        self.model.avgpool = nn.Identity()

    @property
    def backbone_channel(self) -> int:
        return self._backbone_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class ResNet50Backbone(BaseBackbone):
    """ResNet50 Backbone"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        self.model = resnet50(weights=weights)
        self._backbone_channel = 2048
        self.model.fc = nn.Identity()
        self.model.avgpool = nn.Identity()

    @property
    def backbone_channel(self) -> int:
        return self._backbone_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x
