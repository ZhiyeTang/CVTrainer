"""数据相关的测试 fixtures"""
import pytest
import torch
from typing import Dict, Any


@pytest.fixture
def sample_batch(device) -> Dict[str, Any]:
    """返回标准batch样本"""
    return {
        "x": torch.randn(32, 3, 224, 224, device=device),
        "target": {
            "class_id": torch.randint(0, 10, (32,), device=device)
        }
    }


@pytest.fixture
def sample_batch_multiclass(device) -> Dict[str, Any]:
    """返回多分类batch样本（one-hot标签）"""
    num_classes = 10
    return {
        "x": torch.randn(32, 3, 224, 224, device=device),
        "target": {
            "class_id": torch.eye(num_classes)[torch.randint(0, num_classes, (32,))].to(device)
        }
    }


@pytest.fixture
def sample_batch_segmentation(device) -> Dict[str, Any]:
    """返回分割任务batch样本"""
    return {
        "x": torch.randn(8, 3, 256, 256, device=device),
        "target": {
            "mask": torch.randint(0, 21, (8, 256, 256), device=device)
        }
    }


@pytest.fixture
def sample_batch_detection(device) -> Dict[str, Any]:
    """返回检测任务batch样本"""
    batch_size = 4
    return {
        "x": torch.randn(batch_size, 3, 416, 416, device=device),
        "target": {
            "boxes": [
                torch.tensor([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 200.0, 200.0]], device=device),
                torch.tensor([[30.0, 30.0, 70.0, 70.0]], device=device),
                torch.zeros(0, 4, device=device),
                torch.tensor([[5.0, 5.0, 25.0, 25.0]], device=device),
            ],
            "labels": [
                torch.tensor([0, 1], device=device),
                torch.tensor([2], device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.tensor([0], device=device),
            ]
        }
    }


@pytest.fixture
def sample_transforms_config() -> Dict[str, Any]:
    """返回标准transform配置"""
    return {
        "backend": "Albumentations",
        "transforms": [
            {"type": "Resize", "kwargs": {"height": 256, "width": 256}},
            {"type": "HorizontalFlip", "kwargs": {"p": 0.5}},
            {"type": "Normalize", "kwargs": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
        ]
    }


@pytest.fixture
def sample_tensorizer_config() -> Dict[str, Any]:
    """返回标准tensorizer配置"""
    return {
        "type": "cvtrainer.data.tensorizer.Tensorizer",
        "mapping": {
            "x": {
                "type": "cvtrainer.data.tensorizer.ImageTensorizer",
                "kwargs": {"normalize": "imagenet", "dtype": "float"}
            },
            "class_id": {
                "type": "cvtrainer.data.tensorizer.LongTensorizer",
                "kwargs": {}
            }
        }
    }


@pytest.fixture
def sample_dataloader_config() -> Dict[str, Any]:
    """返回标准dataloader配置"""
    return {
        "dataset": {
            "type": "cvtrainer.data.BaseDataAdapter",
            "data_path": "/tmp/test_data"
        },
        "transforms": {
            "backend": "Albumentations",
            "transforms": []
        },
        "tensorizer": {
            "type": "cvtrainer.data.tensorizer.Tensorizer",
            "mapping": {
                "x": {"type": "cvtrainer.data.tensorizer.ImageTensorizer", "kwargs": {}},
                "class_id": {"type": "cvtrainer.data.tensorizer.LongTensorizer", "kwargs": {}}
            }
        },
        "dataloader_args": {
            "batch_size": 32,
            "num_workers": 0,
            "shuffle": True
        }
    }
