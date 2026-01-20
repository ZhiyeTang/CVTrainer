import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from typing import Dict, Any


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Dict[str, Any]
) -> LRScheduler:
    """
    构建 LRScheduler

    Args:
        optimizer: Optimizer 实例
        config: 配置，格式：
            {
                "type": "CosineAnnealingLR",
                "kwargs": {"T_max": 100}
            }

    Returns:
        LRScheduler 实例
    """
    scheduler_type = config["type"]
    kwargs = config.get("kwargs", {})

    scheduler_class = getattr(lr_scheduler, scheduler_type)
    return scheduler_class(optimizer, **kwargs)
