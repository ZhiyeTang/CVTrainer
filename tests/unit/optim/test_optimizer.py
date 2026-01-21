"""优化器模块的单元测试"""
import pytest
import torch
from cvtrainer.optim.optimizer import build_optimizer, _build_layerwise_params


class TestBuildOptimizer:
    """测试 build_optimizer 函数"""

    def test_build_optimizer_adam(self, small_model):
        """测试 Adam 优化器创建"""
        config = {
            "type": "Adam",
            "kwargs": {"lr": 0.001, "weight_decay": 0.0001}
        }
        optimizer = build_optimizer(small_model, config)

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_build_optimizer_sgd(self, small_model):
        """测试 SGD 优化器创建"""
        config = {
            "type": "SGD",
            "kwargs": {"lr": 0.01, "momentum": 0.9, "nesterov": True}
        }
        optimizer = build_optimizer(small_model, config)

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.SGD)

    def test_build_optimizer_adamw(self, small_model):
        """测试 AdamW 优化器创建"""
        config = {
            "type": "AdamW",
            "kwargs": {"lr": 0.001, "weight_decay": 0.01}
        }
        optimizer = build_optimizer(small_model, config)

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_build_optimizer_with_layerwise(self, small_model):
        """测试带分层学习率的优化器"""
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

    def test_build_optimizer_without_layerwise(self, small_model):
        """测试不带分层学习率的优化器"""
        config = {
            "type": "Adam",
            "kwargs": {"lr": 0.001}
        }
        optimizer = build_optimizer(small_model, config)

        assert optimizer is not None
        assert len(list(optimizer.param_groups)) == 1


class TestLayerwiseParams:
    """测试 _build_layerwise_params 函数"""

    def test_layerwise_params_structure(self, small_model):
        """测试分层参数结构"""
        layerwise_config = {
            "backbone": 0.0001,
            "adapter": 0.001,
            "head": 0.01
        }
        default_lr = 0.001

        params = _build_layerwise_params(small_model, layerwise_config, default_lr)

        assert len(params) == 3
        for param_group in params:
            assert "params" in param_group
            assert "lr" in param_group

    def test_layerwise_params_lr_values(self, small_model):
        """测试分层参数学习率值"""
        layerwise_config = {
            "backbone": 0.0001,
            "adapter": 0.001,
            "head": 0.01
        }
        default_lr = 0.001

        params = _build_layerwise_params(small_model, layerwise_config, default_lr)

        lrs = [pg["lr"] for pg in params]
        assert 0.0001 in lrs
        assert 0.001 in lrs
        assert 0.01 in lrs

    def test_layerwise_params_with_missing_keys(self, small_model):
        """测试分层参数缺少配置的情况"""
        layerwise_config = {
            "backbone": 0.0001
        }
        default_lr = 0.001

        params = _build_layerwise_params(small_model, layerwise_config, default_lr)

        lrs = [pg["lr"] for pg in params]
        assert 0.0001 in lrs
        assert default_lr in lrs
