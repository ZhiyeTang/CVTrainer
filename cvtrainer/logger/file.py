import os
import logging
from .base import BaseLogger
from typing import Dict, Any


class FileLogger(BaseLogger):
    """文件 Logger"""
    
    def __init__(self, log_dir: str = "./logs", filename: str = "train.log"):
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("CVTrainerFile")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(log_dir, filename))
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        self.logger.info(f"Metrics: {metrics}")
    
    def log_epoch_summary(self, summary: Dict[str, Any]):
        self.logger.info(f"Epoch Summary: {summary}")
