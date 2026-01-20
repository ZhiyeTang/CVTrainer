from .base import BaseLogger
from typing import Dict, Any


class WandBLogger(BaseLogger):
    """Weights & Biases Logger"""
    
    def __init__(self, project: str, run_name: str = None, config: Dict = None):
        import wandb
        wandb.init(project=project, name=run_name, config=config)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        import wandb
        wandb.log(metrics)
    
    def log_epoch_summary(self, summary: Dict[str, Any]):
        import wandb
        wandb.log(summary)
    
    def close(self):
        import wandb
        wandb.finish()
