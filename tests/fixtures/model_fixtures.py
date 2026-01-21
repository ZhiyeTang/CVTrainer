"""模型相关的测试 fixtures"""
import pytest
import torch
from cvtrainer.models.base import BaseModel
from cvtrainer.models.backbones import ResNet18Backbone
from cvtrainer.models.adapters import ConvAdapter
from cvtrainer.models.heads import MulticlassClassifier


@pytest.fixture
def sample_backbone():
    """返回标准backbone"""
    return ResNet18Backbone(pretrained=False)


@pytest.fixture
def sample_adapter(sample_backbone):
    """返回标准adapter"""
    return ConvAdapter(sample_backbone.backbone_channel, head_channel=256)


@pytest.fixture
def sample_head(sample_adapter):
    """返回标准head"""
    return MulticlassClassifier(sample_adapter.head_channel, num_classes=10)


@pytest.fixture
def sample_model(sample_backbone, sample_adapter, sample_head):
    """返回标准三段式模型"""
    return BaseModel(sample_backbone, sample_adapter, sample_head)


@pytest.fixture
def small_model(device):
    """返回小规模模型用于快速测试"""
    class SimpleBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.backbone_channel = 64

        def forward(self, x):
            return self.conv(x)

    class SimpleAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(64, 128, 1)
            self.head_channel = 128

        def forward(self, x):
            return self.conv(x)

    class SimpleHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(128, 10)

        def forward(self, x):
            return self.fc(x.mean(dim=[2, 3]))

    return BaseModel(
        backbone=SimpleBackbone(),
        adapter=SimpleAdapter(),
        head=SimpleHead()
    ).to(device)


@pytest.fixture
def model_input(device):
    """返回标准模型输入"""
    return torch.randn(2, 3, 224, 224, device=device)


@pytest.fixture
def classification_output_shape():
    """返回分类任务输出形状信息"""
    return {
        "batch_size": 32,
        "num_classes": 10,
        "expected_shape": (32, 10)
    }


@pytest.fixture
def segmentation_output_shape():
    """返回分割任务输出形状信息"""
    return {
        "batch_size": 8,
        "num_classes": 21,
        "height": 256,
        "width": 256,
        "expected_shape": (8, 21, 256, 256)
    }


@pytest.fixture
def detection_output_info():
    """返回检测任务输出信息"""
    return {
        "batch_size": 4,
        "num_classes": 80,
        "max_boxes_per_image": 100,
        "output_keys": ["boxes", "scores", "labels"]
    }
