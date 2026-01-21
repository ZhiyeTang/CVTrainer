"""Trainer相关的测试 fixtures - 使用 CVTrainer 数据系统"""
import pytest
import torch
from torch.utils.data import DataLoader
from cvtrainer.trainer import Trainer
from cvtrainer.meters import LossMeter, AccuracyMeter
from cvtrainer.loss import CrossEntropyLoss
from tests.fixtures.data_fixtures import MockTensorDataset


@pytest.fixture
def simple_dataloader(device, mock_tensor_dataloader):
    """返回简单的DataLoader使用CVTrainer DataAdapter"""
    return mock_tensor_dataloader(batch_size=10, num_samples=100, num_classes=10)


@pytest.fixture
def sample_trainer(device, simple_dataloader, small_model):
    """返回标准Trainer用于测试"""
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

    return Trainer(
        model=small_model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders={"train": simple_dataloader, "val": simple_dataloader},
        hooks=[],
        meters={"loss": LossMeter(), "accuracy": AccuracyMeter()},
        device=device
    )


@pytest.fixture
def trainer_with_hooks(device, simple_dataloader, small_model):
    """返回带有hooks的Trainer"""
    from cvtrainer.hooks.logger import LoggerHook
    from cvtrainer.hooks.checkpoint import CheckpointHook

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

    logger_hook = LoggerHook(train_log_freq=10, val_log_freq=1)
    checkpoint_hook = CheckpointHook(
        save_dir="/tmp/test_checkpoints",
        save_freq=5,
        save_best=False
    )

    return Trainer(
        model=small_model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders={"train": simple_dataloader, "val": simple_dataloader},
        hooks=[logger_hook, checkpoint_hook],
        meters={"loss": LossMeter(), "accuracy": AccuracyMeter()},
        device=device
    )


@pytest.fixture
def trainer_config():
    """返回标准Trainer配置"""
    return {
        "model": "cvtrainer.models.base.BaseModel",
        "model_args": {
            "num_classes": 10
        },
        "criterion": "cvtrainer.loss.CrossEntropyLoss",
        "optimizer": {
            "type": "Adam",
            "kwargs": {"lr": 0.001}
        },
        "epochs": 10,
        "meters": [
            "loss",
            "accuracy"
        ]
    }


@pytest.fixture
def simple_trainer(device, simple_dataloader, small_model):
    """返回简单Trainer用于快速测试"""
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

    return Trainer(
        model=small_model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders={"train": simple_dataloader},
        hooks=[],
        meters={"loss": LossMeter()},
        device=device
    )


@pytest.fixture
def stage_config_with_meters():
    """返回包含meters的stage配置"""
    return {
        "name": "train",
        "epochs": 10,
        "model": "cvtrainer.models.base.BaseModel",
        "model_args": {"num_classes": 10},
        "criterion": "cvtrainer.loss.CrossEntropyLoss",
        "optimizer": {
            "type": "Adam",
            "kwargs": {"lr": 0.001}
        },
        "meters": [
            "loss",
            "accuracy",
            "f1",
            "miou"
        ]
    }
