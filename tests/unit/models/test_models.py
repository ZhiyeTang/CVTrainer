import pytest
import torch
from cvtrainer.models.base import BaseModel
from cvtrainer.models.backbones import ResNet18Backbone
from cvtrainer.models.adapters import ConvAdapter
from cvtrainer.models.heads import MulticlassClassifier


def test_model_creation():
    backbone = ResNet18Backbone(pretrained=False)
    adapter = ConvAdapter(backbone.backbone_channel, head_channel=256)
    head = MulticlassClassifier(adapter.head_channel, num_classes=10)

    model = BaseModel(backbone, adapter, head)

    assert model.backbone.backbone_channel == 512
    assert adapter.head_channel == 256


def test_model_forward():
    backbone = ResNet18Backbone(pretrained=False)
    adapter = ConvAdapter(backbone.backbone_channel, head_channel=256)
    head = MulticlassClassifier(adapter.head_channel, num_classes=10)

    model = BaseModel(backbone, adapter, head)

    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    assert output.shape == (2, 10)
