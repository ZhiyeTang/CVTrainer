import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .hooks.base import BaseHook
    from .meters.base import BaseMeter


@dataclass
class Stage:
    """训练阶段的所有组件"""

    criterion: torch.nn.Module
    optimizer: Optimizer
    dataloaders: Dict[str, DataLoader]
    model: torch.nn.Module
    hooks: List["BaseHook"]
    meters: Dict[str, "BaseMeter"]
