import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from .base import BaseHook
from typing import Dict, Any


class DDPHook(BaseHook):
    """DDP 训练 Hook - 包含 DDP 初始化和 meter 同步"""
    
    def __init__(
        self,
        num_gpus: int = 1,
        find_unused_parameters: bool = False,
    ):
        self.num_gpus = num_gpus
        self.find_unused_parameters = find_unused_parameters
        self.rank = 0
        self.world_size = 1
        self.samplers = {}
        self.distributed = False
        self.distributed_model = None
        self.original_model = None
    
    def before_train(self, stage: Dict[str, Any]):
        """初始化 DDP"""
        if self.num_gpus > 1:
            self.distributed = True
            self._setup_ddp()
            self._wrap_model(stage["trainer"])
            self._replace_samplers(stage["trainer"])
            
            stage["context"].set("rank", self.rank)
            stage["context"].set("world_size", self.world_size)
            stage["context"].set("distributed", self.distributed)
    
    def _setup_ddp(self):
        """设置 DDP"""
        if not dist.is_available():
            raise RuntimeError("DDP is not available")
        
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
        
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            world_size=self.num_gpus,
            rank=self.rank,
        )
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
    
    def _wrap_model(self, trainer):
        """使用 DDP 包装模型"""
        if self.distributed:
            self.original_model = trainer.model
            self.distributed_model = DDP(
                trainer.model,
                device_ids=[self.rank],
                find_unused_parameters=self.find_unused_parameters,
            )
            trainer.model = self.distributed_model
    
    def _replace_samplers(self, trainer):
        """用 DistributedSampler 替换默认 sampler"""
        if self.distributed:
            for split, dataloader in trainer.dataloaders.items():
                dataset = dataloader.dataset
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=split == "train",
                )
                dataloader.sampler = sampler
                self.samplers[split] = sampler
    
    def before_epoch(self, stage: Dict[str, Any]):
        """每个 epoch 开始前设置 sampler 的 epoch"""
        if self.distributed:
            for sampler in self.samplers.values():
                sampler.set_epoch(stage["epoch"])
    
    def after_eval(self, stage: Dict[str, Any]):
        """评估结束后同步所有进程的 meter 结果"""
        if self.distributed:
            self._sync_meters(stage["trainer"])
    
    def _sync_meters(self, trainer):
        """同步 meters"""
        meters = trainer.meters
        for meter in meters.values():
            self._sync_meter(meter)
    
    def _sync_meter(self, meter):
        """同步单个 meter"""
        if hasattr(meter, "correct"):
            if isinstance(meter.correct, list):
                meter.correct = [self._sync_tensor(torch.tensor(c)) for c in meter.correct]
            else:
                meter.correct = self._sync_tensor(torch.tensor(meter.correct))
        if hasattr(meter, "total"):
            meter.total = self._sync_tensor(torch.tensor(meter.total))
        if hasattr(meter, "sum"):
            meter.sum = self._sync_tensor(torch.tensor(meter.sum))
        if hasattr(meter, "count"):
            meter.count = self._sync_tensor(torch.tensor(meter.count))
    
    def _sync_tensor(self, tensor: torch.Tensor) -> int:
        """同步 tensor"""
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()
    
    def after_train(self, stage: Dict[str, Any]):
        """清理 DDP"""
        if self.distributed:
            if self.distributed_model is not None:
                stage["trainer"].model = self.original_model
            dist.destroy_process_group()
