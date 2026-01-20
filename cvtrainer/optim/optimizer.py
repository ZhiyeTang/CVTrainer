import torch
from torch.optim import Optimizer
from typing import Dict, Any, Union


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> Optimizer:
    """
    构建 Optimizer，支持分层学习率

    Args:
        model: 模型
        config: 配置，格式：
            {
                "type": "Adam",
                "kwargs": {"lr": 0.001},
                "layerwise": {
                    "backbone": 0.001,
                    "adapter": 0.01,
                    "head": 0.1
                }
            }

    Returns:
        Optimizer 实例
    """
    optimizer_type = config["type"]
    kwargs = config.get("kwargs", {})
    layerwise_config = config.get("layerwise", None)

    if layerwise_config:
        params = _build_layerwise_params(model, layerwise_config, kwargs["lr"])
    else:
        params = model.parameters()

    optimizer_class = getattr(torch.optim, optimizer_type)
    return optimizer_class(params, **kwargs)


def _build_layerwise_params(
    model: torch.nn.Module, layerwise_config: Dict[str, float], default_lr: float
):
    """构建分层学习率参数"""
    params = []

    backbone_lr = layerwise_config.get("backbone", default_lr)
    adapter_lr = layerwise_config.get("adapter", default_lr)
    head_lr = layerwise_config.get("head", default_lr)

    params.append({"params": model.backbone.parameters(), "lr": backbone_lr})
    params.append({"params": model.adapter.parameters(), "lr": adapter_lr})
    params.append({"params": model.head.parameters(), "lr": head_lr})

    return params
