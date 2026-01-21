"""Trainer 模块的单元测试"""
import pytest
import torch
from cvtrainer.trainer import Trainer
from cvtrainer.meters import LossMeter, AccuracyMeter
from cvtrainer.loss import CrossEntropyLoss


class TestTrainerInitialization:
    """测试 Trainer 初始化"""

    def test_trainer_initialization(self, device, simple_dataloader, small_model):
        """测试基本初始化"""
        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

        trainer = Trainer(
            model=small_model,
            criterion=criterion,
            optimizer=optimizer,
            dataloaders={"train": simple_dataloader},
            hooks=[],
            meters={"loss": LossMeter()},
            device=device
        )

        assert trainer.model is not None
        assert trainer.criterion is not None
        assert trainer.optimizer is not None
        assert trainer.epoch == 0
        assert trainer.step == 0

    def test_trainer_initialization_with_meters(self, device, simple_dataloader, small_model):
        """测试带 meters 的初始化"""
        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

        trainer = Trainer(
            model=small_model,
            criterion=criterion,
            optimizer=optimizer,
            dataloaders={"train": simple_dataloader},
            hooks=[],
            meters={"loss": LossMeter(), "accuracy": AccuracyMeter()},
            device=device
        )

        assert "loss" in trainer.meters
        assert "accuracy" in trainer.meters

    def test_trainer_initialization_with_hooks(self, device, simple_dataloader, small_model):
        """测试带 hooks 的初始化"""
        from cvtrainer.hooks.logger import LoggerHook

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        logger_hook = LoggerHook(train_log_freq=10, val_log_freq=1)

        trainer = Trainer(
            model=small_model,
            criterion=criterion,
            optimizer=optimizer,
            dataloaders={"train": simple_dataloader},
            hooks=[logger_hook],
            meters={"loss": LossMeter()},
            device=device
        )

        assert len(trainer.hooks) == 1


class TestTrainerStateDict:
    """测试 Trainer 状态保存和加载"""

    def test_state_dict(self, sample_trainer):
        """测试状态保存"""
        state = sample_trainer.state_dict()

        assert "model" in state
        assert "optimizer" in state
        assert "epoch" in state
        assert "step" in state

    def test_load_state_dict(self, sample_trainer):
        """测试状态加载"""
        sample_trainer.epoch = 5
        sample_trainer.step = 100

        state = sample_trainer.state_dict()
        sample_trainer.load_state_dict(state)

        assert sample_trainer.epoch == 5
        assert sample_trainer.step == 100


class TestFlattenMeters:
    """测试 _flatten_meters 方法"""

    def test_flatten_single_value(self, sample_trainer):
        """测试展平单个值"""
        result = sample_trainer._flatten_meters(sample_trainer.meters)

        assert isinstance(result, dict)
        assert "loss" in result
        assert isinstance(result["loss"], float)


class TestUpdateMeters:
    """测试 _update_meters 方法"""

    def test_update_loss_meter(self, sample_trainer):
        """测试更新 LossMeter"""
        output = torch.randn(10, 10)
        target = torch.randint(0, 10, (10,))
        loss = 0.5

        sample_trainer._update_meters(output, target, loss)

    def test_update_accuracy_meter(self, sample_trainer):
        """测试更新 AccuracyMeter"""
        output = torch.randn(10, 10)
        target = torch.randint(0, 10, (10,))

        sample_trainer.meters["accuracy"] = AccuracyMeter()
        sample_trainer._update_meters(output, target, 0.5)

    def test_update_with_onehot_target(self, sample_trainer):
        """测试更新带 one-hot 标签的情况"""
        output = torch.randn(10, 10)
        target = torch.eye(10)[torch.randint(0, 10, (10,))]

        sample_trainer.meters["accuracy"] = AccuracyMeter()
        sample_trainer._update_meters(output, target, 0.5)


class TestCallHooks:
    """测试 call_hooks 方法"""

    def test_call_hooks_nonexistent(self, sample_trainer):
        """测试调用不存在的 hook"""
        sample_trainer.call_hooks("before_nonexistent")

    def test_call_hooks_before_step(self, sample_trainer):
        """测试调用 before_step hook"""
        sample_trainer.call_hooks("before_step")


class TestContext:
    """测试上下文管理"""

    def test_context_set_and_get(self, sample_trainer):
        """测试上下文设置和获取"""
        sample_trainer.context.set("test_key", "test_value")
        assert sample_trainer.context.get("test_key") == "test_value"

    def test_context_loss(self, sample_trainer):
        """测试 loss 上下文"""
        sample_trainer.context.set("loss", 0.5)
        loss = sample_trainer.context.get("loss")
        assert loss == 0.5
