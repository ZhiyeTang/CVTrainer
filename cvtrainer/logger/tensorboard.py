import os
from .base import BaseLogger
from typing import Dict, Any


class TensorBoardLogger(BaseLogger):
    """TensorBoard Logger"""
    
    def __init__(self, log_dir: str = "./runs"):
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        step = metrics.get("step", 0)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
    
    def log_epoch_summary(self, summary: Dict[str, Any]):
        step = summary.get("epoch", 0)
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"epoch/{key}", value, step)
    
    def close(self):
        self.writer.close()
