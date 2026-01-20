import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Any
from .hooks.base import BaseHook
from .meters.base import BaseMeter
from .context import HookContext


class Trainer:
    """
    极简训练器类 - 只负责最基础的业务过程
    
    核心属性：
    - model, criterion, optimizer, dataloaders: 训练资产
    - hooks: 扩展机制，所有训练技巧通过 hook 实现
    - meters: 指标计算
    - context: hook 间通信上下文
    
    核心方法：
    - train(): 训练循环
    - train_step(): 单步训练
    - eval(): 评估
    - eval_step(): 单步评估
    - state_dict(): 获取所有资产状态
    - load_state_dict(): 加载状态
    - call_hooks(): 调用 hooks
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        dataloaders: Dict[str, DataLoader],
        hooks: List[BaseHook] = None,
        meters: Dict[str, BaseMeter] = None,
        device: torch.device = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.hooks = hooks or []
        self.meters = meters or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.context = HookContext()
        
        self.epoch = 0
        self.step = 0
        self.best_metric = None
        self.best_metric_name = None
        
        self.model.to(self.device)
        self.criterion.to(self.device)
    
    def call_hooks(self, hook_name: str, **kwargs):
        """调用指定的 hook 方法"""
        kwargs["context"] = self.context
        kwargs["trainer"] = self
        
        for hook in self.hooks:
            method = getattr(hook, hook_name, None)
            if method:
                method(stage=kwargs)
    
    def train(self, epochs: int, resume: bool = False):
        """训练循环"""
        phase = "train"
        self.call_hooks("before_train", epoch=0, phase=phase)
        
        start_epoch = self.epoch if resume else 0
        for epoch in range(start_epoch, epochs):
            self.epoch = epoch
            
            if "train" in self.dataloaders:
                phase = "train"
                self.context.set("phase", phase)
                self.call_hooks("before_epoch", epoch=epoch, phase=phase)
                self._train_epoch()
                self.call_hooks("after_epoch", epoch=epoch, phase=phase)
            
            if "val" in self.dataloaders:
                phase = "val"
                self.context.set("phase", phase)
                self.call_hooks("before_eval", epoch=epoch, phase=phase)
                self.eval("val")
                self.call_hooks("after_eval", epoch=epoch, phase=phase)
        
        self.call_hooks("after_train", epoch=epochs, phase="train")
    
    def _train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        dataloader = self.dataloaders["train"]
        
        for batch_idx, batch in enumerate(dataloader):
            self.step = self.epoch * len(dataloader) + batch_idx
            
            self.call_hooks("before_step", batch=batch)
            self.train_step(batch)
            self.call_hooks("after_step", batch=batch)
    
    def train_step(self, batch: Dict[str, torch.Tensor]):
        """单步训练"""
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "target"}
        targets = {k: v.to(self.device) for k, v in batch["target"].items()}
        
        self.call_hooks("before_forward", batch=batch)
        
        outputs = self.model(inputs["x"])
        loss = self.criterion(outputs, targets["class_id"])
        
        self.context.set("loss", loss.item())
        self.context.set("outputs", outputs)
        
        self.call_hooks("after_forward", batch=batch, outputs=outputs, loss=loss.item())
        self.call_hooks("before_backward", batch=batch, outputs=outputs, loss=loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        
        self.call_hooks("after_backward", batch=batch, outputs=outputs, loss=loss.item())
        self.call_hooks("before_optimize", batch=batch, outputs=outputs, loss=loss.item())
        
        self.optimizer.step()
        
        self.call_hooks("after_optimize", batch=batch, outputs=outputs, loss=loss.item())
        
        self._update_meters(outputs, targets["class_id"], loss.item())
        
        meters_values = {name: meter.get_value() for name, meter in self.meters.items()}
        self.context.set("meters", meters_values)
    
    def _update_meters(self, outputs: torch.Tensor, targets: torch.Tensor, loss: float):
        """更新 meters"""
        for meter in self.meters.values():
            if hasattr(meter, "update"):
                if hasattr(meter, "__class__") and "Accuracy" in meter.__class__.__name__:
                    meter.update(outputs, targets)
                elif hasattr(meter, "__class__") and "Loss" in meter.__class__.__name__:
                    meter.update(loss)
    
    def eval(self, split: str = "val"):
        """评估"""
        self.model.eval()
        dataloader = self.dataloaders[split]
        
        for meter in self.meters.values():
            meter.reset()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                self.eval_step(batch)
        
        meters_values = {name: meter.get_value() for name, meter in self.meters.items()}
        self.context.set("meters", meters_values)
    
    def eval_step(self, batch: Dict[str, torch.Tensor]):
        """单步评估"""
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "target"}
        targets = {k: v.to(self.device) for k, v in batch["target"].items()}
        
        outputs = self.model(inputs["x"])
        loss = self.criterion(outputs, targets["class_id"])
        
        self.context.set("loss", loss.item())
        
        self._update_meters(outputs, targets["class_id"], loss.item())
    
    def state_dict(self) -> Dict[str, Any]:
        """递归返回所有资产的 state_dict"""
        meters_state = {}
        for name, meter in self.meters.items():
            if hasattr(meter, "state_dict"):
                meters_state[name] = meter.state_dict()
            else:
                meters_state[name] = meter.__dict__
        
        hooks_state = {}
        for i, hook in enumerate(self.hooks):
            hook_name = f"hook_{i}_{hook.__class__.__name__}"
            if hasattr(hook, "state_dict"):
                hooks_state[hook_name] = hook.state_dict()
            elif hasattr(hook, "__dict__"):
                hooks_state[hook_name] = hook.__dict__
        
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "meters": meters_state,
            "hooks": hooks_state,
            "epoch": self.epoch,
            "step": self.step,
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
        }
    
    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
        load_hooks: bool = False,
        load_optim: bool = False,
    ):
        """递归加载所有资产的 state_dict"""
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"], strict=strict)
        
        self.epoch = state_dict.get("epoch", 0)
        self.step = state_dict.get("step", 0)
        self.best_metric = state_dict.get("best_metric", None)
        self.best_metric_name = state_dict.get("best_metric_name", None)
        
        if load_optim and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        
        if "hooks" in state_dict:
            for hook_name, hook_state in state_dict["hooks"].items():
                parts = hook_name.split("_")
                if len(parts) >= 3:
                    try:
                        hook_index = int(parts[1])
                        if hook_index < len(self.hooks):
                            hook = self.hooks[hook_index]
                            if "Logger" in hook.__class__.__name__:
                                if hasattr(hook, "load_state_dict"):
                                    hook.load_state_dict(hook_state)
                                else:
                                    hook.__dict__.update(hook_state)
                            
                            if load_hooks:
                                if "Logger" not in hook.__class__.__name__:
                                    if hasattr(hook, "load_state_dict"):
                                        hook.load_state_dict(hook_state)
                                    else:
                                        hook.__dict__.update(hook_state)
                    except (ValueError, IndexError):
                        pass
