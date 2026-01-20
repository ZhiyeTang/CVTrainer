import logging
from .base import BaseLogger
from typing import Dict, Any


class ConsoleLogger(BaseLogger):
    """控制台 Logger"""
    
    def __init__(self, level: str = "INFO"):
        self.level = level
        self.logger = logging.getLogger("CVTrainer")
        self.logger.setLevel(getattr(logging, level))
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        self.logger.info(f"Metrics: {metrics}")
    
    def log_epoch_summary(self, summary: Dict[str, Any]):
        self.logger.info(f"Epoch Summary: {summary}")
