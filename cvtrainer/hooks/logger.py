from .base import BaseHook
from typing import Dict, Any


class LoggerHook(BaseHook):
    """日志记录 Hook"""
    
    def __init__(
        self,
        train_log_freq: int = 10,
        val_log_freq: int = 1,
    ):
        self.train_log_freq = train_log_freq
        self.val_log_freq = val_log_freq
        self.loggers = []
        self.logged_values = {}
    
    def add_logger(self, logger):
        """添加 logger"""
        self.loggers.append(logger)
    
    def after_step(self, stage: Dict[str, Any]):
        """每步后记录日志"""
        context = stage["context"]
        phase = context.get("phase", "train")
        freq = self.train_log_freq if phase == "train" else self.val_log_freq
        
        if stage["step"] % freq == 0:
            metrics = {
                "epoch": stage["epoch"],
                "step": stage["step"],
                "phase": phase,
                "loss": context.get("loss", 0),
                **context.get("meters", {}),
            }
            for logger in self.loggers:
                logger.log_metrics(metrics)
            self.logged_values["last_step"] = stage["step"]
    
    def after_epoch(self, stage: Dict[str, Any]):
        """每个 epoch 后记录总结"""
        context = stage["context"]
        summary = {
            "epoch": stage["epoch"],
            "phase": context.get("phase", "train"),
            **context.get("meters", {}),
        }
        for logger in self.loggers:
            logger.log_epoch_summary(summary)
    
    def state_dict(self) -> Dict[str, Any]:
        """获取 state_dict"""
        return {
            "logged_values": self.logged_values,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载 state_dict"""
        if "logged_values" in state_dict:
            self.logged_values = state_dict["logged_values"]
