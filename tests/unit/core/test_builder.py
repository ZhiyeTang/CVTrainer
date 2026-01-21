"""Builder 模块的单元测试"""
import pytest
import torch
from cvtrainer.builder import (
    build_stage,
    build_trainer,
    build_loggers,
    _build_model,
    _build_criterion,
    _build_dataloaders,
    _build_hooks,
    _build_meters,
)
from cvtrainer.optim.optimizer import build_optimizer
from cvtrainer.optim.scheduler import build_scheduler


class TestBuildOptimizer:
    """测试 build_optimizer 函数"""

    def test_build_optimizer_adam(self, small_model):
        """测试 Adam 优化器创建"""
        config = {
            "type": "Adam",
            "kwargs": {"lr": 0.001}
        }
        optimizer = build_optimizer(small_model, config)

        assert optimizer is not None
        assert isinstance(optimizer, type(torch.optim.Adam({})))
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_build_optimizer_sgd(self, small_model):
        """测试 SGD 优化器创建"""
        config = {
            "type": "SGD",
            "kwargs": {"lr": 0.01, "momentum": 0.9}
        }
        optimizer = build_optimizer(small_model, config)

        assert optimizer is not None
        assert isinstance(optimizer, type(torch.optim.SGD({}, lr=0.01)))

    def test_build_optimizer_with_layerwise(self, small_model):
        """测试带分层学习率的优化器创建"""
        config = {
            "type": "Adam",
            "kwargs": {"lr": 0.001},
            "layerwise": {
                "backbone": 0.0001,
                "adapter": 0.001,
                "head": 0.01
            }
        }
        optimizer = build_optimizer(small_model, config)

        assert optimizer is not None
        assert len(optimizer.param_groups) == 3


class TestBuildScheduler:
    """测试 build_scheduler 函数"""

    def test_build_scheduler_cosine(self, simple_trainer):
        """测试 CosineAnnealingLR 调度器创建"""
        config = {
            "type": "CosineAnnealingLR",
            "kwargs": {"T_max": 100}
        }
        scheduler = build_scheduler(simple_trainer.optimizer, config)

        assert scheduler is not None
        assert isinstance(scheduler, type(torch.optim.lr_scheduler.CosineAnnealingLR(simple_trainer.optimizer, T_max=100)))

    def test_build_scheduler_step(self, simple_trainer):
        """测试 StepLR 调度器创建"""
        config = {
            "type": "StepLR",
            "kwargs": {"step_size": 30, "gamma": 0.1}
        }
        scheduler = build_scheduler(simple_trainer.optimizer, config)

        assert scheduler is not None


class TestBuildMeters:
    """测试 _build_meters 函数"""

    def test_build_meters_basic(self, stage_config_with_meters):
        """测试基本 meter 构建"""
        meters = _build_meters(stage_config_with_meters)

        assert "loss" in meters
        assert isinstance(meters["loss"].__class__.__name__, str)

    def test_build_meters_with_accuracy(self):
        """测试带 AccuracyMeter 的构建"""
        config = {
            "meters": ["accuracy"]
        }
        meters = _build_meters(config)

        assert "accuracy" in meters

    def test_build_meters_with_f1(self):
        """测试带 F1Meter 的构建"""
        config = {
            "meters": ["f1"]
        }
        meters = _build_meters(config)

        assert "f1" in meters

    def test_build_meters_with_map(self):
        """测试带 DetectionMapMeter 的构建"""
        config = {
            "meters": ["map"]
        }
        meters = _build_meters(config)

        assert "map" in meters

    def test_build_meters_with_miou(self):
        """测试带 SegmentationIoUMeter 的构建"""
        config = {
            "meters": ["miou"]
        }
        meters = _build_meters(config)

        assert "miou" in meters


class TestBuildHooks:
    """测试 _build_hooks 函数"""

    def test_build_hooks_empty(self):
        """测试空 hooks 构建"""
        config = {"hooks": []}
        hooks = _build_hooks(config)

        assert hooks == []

    def test_build_hooks_with_logger(self):
        """测试带 LoggerHook 的构建"""
        config = {
            "hooks": [
                {"type": "cvtrainer.hooks.logger.LoggerHook", "kwargs": {}}
            ]
        }
        hooks = _build_hooks(config)

        assert len(hooks) == 1
        assert hooks[0].__class__.__name__ == "LoggerHook"

    def test_build_hooks_multiple(self):
        """测试多个 hooks 的构建"""
        config = {
            "hooks": [
                {"type": "cvtrainer.hooks.logger.LoggerHook", "kwargs": {}},
                {"type": "cvtrainer.hooks.checkpoint.CheckpointHook", "kwargs": {}}
            ]
        }
        hooks = _build_hooks(config)

        assert len(hooks) == 2
