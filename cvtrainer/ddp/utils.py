import torch.distributed as dist


def setup_ddp():
    """设置 DDP"""
    if not dist.is_available():
        return

    if dist.is_initialized():
        return

    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")


def cleanup_ddp():
    """清理 DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """是否为主进程"""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """获取 world size"""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取 rank"""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()
