from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseHook(ABC):
    """Hook 基类，定义16个调用点"""

    def before_train(self, stage: Dict[str, Any]):
        """训练开始前调用"""
        pass

    def after_train(self, stage: Dict[str, Any]):
        """训练结束后调用"""
        pass

    def before_epoch(self, stage: Dict[str, Any]):
        """每个 epoch 开始前调用"""
        pass

    def after_epoch(self, stage: Dict[str, Any]):
        """每个 epoch 结束后调用"""
        pass

    def before_step(self, stage: Dict[str, Any]):
        """每个 step 开始前调用"""
        pass

    def after_step(self, stage: Dict[str, Any]):
        """每个 step 结束后调用"""
        pass

    def before_forward(self, stage: Dict[str, Any]):
        """forward 前调用"""
        pass

    def after_forward(self, stage: Dict[str, Any]):
        """forward 后调用"""
        pass

    def before_backward(self, stage: Dict[str, Any]):
        """backward 前调用"""
        pass

    def after_backward(self, stage: Dict[str, Any]):
        """backward 后调用"""
        pass

    def before_optimize(self, stage: Dict[str, Any]):
        """optimize 前调用"""
        pass

    def after_optimize(self, stage: Dict[str, Any]):
        """optimize 后调用"""
        pass

    def before_eval(self, stage: Dict[str, Any]):
        """评估开始前调用"""
        pass

    def after_eval(self, stage: Dict[str, Any]):
        """评估结束后调用"""
        pass

    def before_save(self, stage: Dict[str, Any]):
        """保存前调用"""
        pass

    def after_save(self, stage: Dict[str, Any]):
        """保存后调用"""
        pass
