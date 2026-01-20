from tqdm import tqdm
from .base import BaseHook
from typing import Dict, Any


class ProgressBarHook(BaseHook):
    """进度条 Hook - 实时显示训练进度"""
    
    def __init__(self):
        self.pbar = None
        self.train_pbar = None
        self.val_pbar = None
        self.train_meters = {}
        self.val_meters = {}
    
    def before_train(self, stage: Dict[str, Any]):
        """初始化进度条"""
        trainer = stage["trainer"]
        self.train_meters = {name: None for name in trainer.meters.keys()}
        self.val_meters = {name: None for name in trainer.meters.keys()}
    
    def before_epoch(self, stage: Dict[str, Any]):
        """每个 epoch 开始前创建进度条"""
        trainer = stage["trainer"]
        phase = stage["phase"]
        dataloader = trainer.dataloaders[phase]
        
        if phase == "train":
            self.train_pbar = tqdm(
                total=len(dataloader),
                desc=f"Epoch {stage['epoch']} [Train]",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            self.val_pbar = tqdm(
                total=len(dataloader),
                desc=f"Epoch {stage['epoch']} [Val]",
                leave=True,
                dynamic_ncols=True,
            )
    
    def after_step(self, stage: Dict[str, Any]):
        """每步后更新进度条"""
        context = stage["context"]
        phase = context.get("phase", "train")
        loss = context.get("loss", 0)
        meters = context.get("meters", {})
        
        if phase == "train" and self.train_pbar is not None:
            first_meter_name = list(self.train_meters.keys())[0] if self.train_meters else None
            first_meter_value = meters.get(first_meter_name, 0) if first_meter_name else 0
            
            postfix = {"loss": f"{loss:.4f}"}
            if first_meter_name:
                postfix[first_meter_name] = f"{first_meter_value:.4f}"
            
            self.train_pbar.set_postfix(postfix)
            self.train_pbar.update(1)
    
    def after_eval(self, stage: Dict[str, Any]):
        """评估结束后更新 val 进度条"""
        context = stage["context"]
        loss = context.get("loss", 0)
        meters = context.get("meters", {})
        
        if self.val_pbar is not None:
            first_meter_name = list(self.val_meters.keys())[0] if self.val_meters else None
            first_meter_value = meters.get(first_meter_name, 0) if first_meter_name else 0
            
            postfix = {"loss": f"{loss:.4f}"}
            if first_meter_name:
                postfix[first_meter_name] = f"{first_meter_value:.4f}"
            
            self.val_pbar.set_postfix(postfix)
            self.val_pbar.update(1)
    
    def after_epoch(self, stage: Dict[str, Any]):
        """每个 epoch 后关闭进度条"""
        phase = stage["phase"]
        
        if phase == "train" and self.train_pbar is not None:
            self.train_pbar.close()
            self.train_pbar = None
        elif phase == "val" and self.val_pbar is not None:
            self.val_pbar.close()
            self.val_pbar = None
