import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from .utils import setup_ddp, cleanup_ddp


def launch_ddp(train_fn, world_size: int, *args, **kwargs):
    """
    启动 DDP 训练

    Args:
        train_fn: 训练函数
        world_size: GPU 数量
        args, kwargs: 传递给 train_fn 的参数
    """
    mp.spawn(
        train_wrapper,
        args=(train_fn, world_size, args, kwargs),
        nprocs=world_size,
        join=True,
    )


def train_wrapper(rank, train_fn, world_size, args, kwargs):
    """DDP 训练包装器"""
    setup_ddp()

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    train_fn(rank, world_size, device, *args, **kwargs)

    cleanup_ddp()
