import torch
import torch.nn as nn
from cvtrainer.models.base import BaseModel
from cvtrainer.models.backbones import ResNet18Backbone
from cvtrainer.models.adapters import IdentityAdapter
from cvtrainer.models.heads import MulticlassClassifier


class CIFAR10Model(BaseModel):
    """CIFAR-10分类模型：ResNet18 + Identity + Classifier"""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        backbone = ResNet18Backbone(pretrained=False)
        adapter = IdentityAdapter(backbone.backbone_channel)
        head = MulticlassClassifier(adapter.head_channel, num_classes, dropout_rate)
        super().__init__(backbone, adapter, head)
