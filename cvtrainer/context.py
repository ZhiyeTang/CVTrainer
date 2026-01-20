from typing import Dict, Any


class HookContext:
    """
    Hook 上下文类 - 用于不同 hook 之间相互通信
    
    允许 hook 之间共享状态和信息，例如：
    - CheckpointHook 可以从 HookContext 获取当前的 best_metric
    - DDPHook 可以在 HookContext 中存储 rank 和 world_size
    - ProgressBarHook 可以从 HookContext 获取当前的 loss 和 meters
    """
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any):
        """设置上下文值"""
        self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取上下文值"""
        return self._data.get(key, default)
    
    def has(self, key: str) -> bool:
        """检查是否存在某个键"""
        return key in self._data
    
    def remove(self, key: str):
        """移除上下文值"""
        if key in self._data:
            del self._data[key]
    
    def clear(self):
        """清空所有上下文"""
        self._data.clear()
    
    def update(self, data: Dict[str, Any]):
        """批量更新上下文"""
        self._data.update(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._data.copy()
    
    def __repr__(self) -> str:
        return f"HookContext({self._data})"
