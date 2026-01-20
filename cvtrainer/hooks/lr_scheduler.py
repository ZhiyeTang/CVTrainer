import torch.optim.lr_scheduler as lr_scheduler
from .base import BaseHook
from typing import Dict, Any, Type
import importlib


class LRSchedulerHook(BaseHook):
    """学习率调度 Hook"""

    def __init__(self, scheduler_type: str, scheduler_kwargs: Dict[str, Any]):
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler = None

    def setup_scheduler(self, optimizer):
        """根据 optimizer 创建 scheduler"""
        scheduler_module = importlib.import_module("torch.optim.lr_scheduler")
        scheduler_class = getattr(scheduler_module, self.scheduler_type)
        self.scheduler = scheduler_class(optimizer, **self.scheduler_kwargs)

    def after_epoch(self, stage: Dict[str, Any]):
        """每个 epoch 结束后更新学习率"""
        if self.scheduler:
            self.scheduler.step()
