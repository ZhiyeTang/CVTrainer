import torch
from .base import BaseHook
from typing import Dict, Any


class EMAHook(BaseHook):
    """指数移动平均 Hook"""
    
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.ema_model = None
        self.shadow = {}
        self.original_model = None
    
    def before_train(self, stage: Dict[str, Any]):
        """初始化 EMA 模型"""
        model = stage["model"]
        if hasattr(model, "module"):
            model = model.module
        
        self.ema_model = type(model)(model.backbone, model.adapter, model.head)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.eval()
        self.ema_model.to(model.device)
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def after_step(self, stage: Dict[str, Any]):
        """更新 EMA"""
        model = stage["model"]
        if hasattr(model, "module"):
            model = model.module
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        
        with torch.no_grad():
            for name, param in self.ema_model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])
    
    def before_eval(self, stage: Dict[str, Any]):
        """评估前使用 EMA 模型"""
        trainer = stage["trainer"]
        model = trainer.model
        
        self.original_model = model
        
        if hasattr(model, "module"):
            original = model.module
        else:
            original = model
        
        trainer.model = self.ema_model
        
        stage["context"].set("original_model", original)
    
    def after_eval(self, stage: Dict[str, Any]):
        """评估后恢复原始模型"""
        trainer = stage["trainer"]
        context = stage["context"]
        
        original = context.get("original_model")
        if original is not None:
            trainer.model = original
            context.remove("original_model")
