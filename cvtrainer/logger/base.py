from typing import Dict, Any


class BaseLogger:
    """Logger 基类"""
    
    def log_metrics(self, metrics: Dict[str, Any]):
        raise NotImplementedError
    
    def log_epoch_summary(self, summary: Dict[str, Any]):
        raise NotImplementedError
    
    def close(self):
        pass
