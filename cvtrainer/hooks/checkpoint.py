import os
import torch
import hashlib
import json
from datetime import datetime
from .base import BaseHook
from typing import Dict, Any


class CheckpointHook(BaseHook):
    """检查点保存 Hook"""
    
    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_freq: int = 10,
        save_best: bool = True,
        metric_name: str = "val_accuracy",
        mode: str = "max",
    ):
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        self.best_value = None
        self.config_hash = None
    
    def before_train(self, stage: Dict[str, Any]):
        """创建保存目录，计算配置哈希"""
        os.makedirs(self.save_dir, exist_ok=True)
        
        context = stage["context"]
        config = context.get("config", {})
        config_str = json.dumps(config, sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        
        context.set("best_value", self.best_value)
    
    def after_epoch(self, stage: Dict[str, Any]):
        """每个 epoch 后保存检查点"""
        epoch = stage["epoch"]
        phase = stage["phase"]
        
        if phase == "val" and (epoch + 1) % self.save_freq == 0:
            self._save_checkpoint(stage)
        
        if self.save_best and phase == "val":
            self._save_best_checkpoint(stage)
    
    def _save_checkpoint(self, stage: Dict[str, Any]):
        """保存定期检查点"""
        trainer = stage["trainer"]
        epoch = stage["epoch"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch_{epoch + 1}_{timestamp}_{self.config_hash}.pth"
        save_path = os.path.join(self.save_dir, filename)
        
        self._save(trainer, save_path, epoch)
    
    def _save_best_checkpoint(self, stage: Dict[str, Any]):
        """保存最佳模型"""
        trainer = stage["trainer"]
        context = stage["context"]
        meters = context.get("meters", {})
        
        if self.metric_name not in meters:
            return
        
        current_value = meters[self.metric_name]
        best_value = context.get("best_value")
        
        if best_value is None:
            self.best_value = current_value
            context.set("best_value", self.best_value)
            self._save_best(trainer, stage["epoch"])
        elif (self.mode == "max" and current_value > best_value) or \
             (self.mode == "min" and current_value < best_value):
            self.best_value = current_value
            context.set("best_value", self.best_value)
            self._save_best(trainer, stage["epoch"])
    
    def _save_best(self, trainer, epoch: int):
        """保存最佳模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_model_{timestamp}_{self.config_hash}.pth"
        save_path = os.path.join(self.save_dir, filename)
        self._save(trainer, save_path, epoch, is_best=True)
    
    def _save(self, trainer, save_path: str, epoch: int, is_best: bool = False):
        """保存检查点"""
        state_dict = trainer.state_dict()
        state_dict["is_best"] = is_best
        torch.save(state_dict, save_path)
    
    def before_save(self, stage: Dict[str, Any]):
        """保存前调用"""
        pass
    
    def after_save(self, stage: Dict[str, Any]):
        """保存后调用"""
        pass
