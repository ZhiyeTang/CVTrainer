import importlib
from typing import Any, Dict
from .core import Stage
from .trainer import Trainer


def import_module_str(module_str: str) -> Any:
    """通过 importlib 导入模块并获取类/函数"""
    parts = module_str.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_stage(stage_config: Dict[str, Any]) -> Stage:
    """构建 Stage 实例"""
    model = _build_model(stage_config)
    criterion = _build_criterion(stage_config)
    optimizer = _build_optimizer(stage_config, model)
    dataloaders = _build_dataloaders(stage_config)
    hooks = _build_hooks(stage_config)
    meters = _build_meters(stage_config)

    return Stage(
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=dataloaders,
        model=model,
        hooks=hooks,
        meters=meters,
    )


def build_trainer(stage_config: Dict[str, Any], loggers: list = None) -> Trainer:
    """构建 Trainer"""
    model = _build_model(stage_config)
    criterion = _build_criterion(stage_config)
    optimizer = _build_optimizer(stage_config, model)
    dataloaders = _build_dataloaders(stage_config)
    hooks = _build_hooks(stage_config)
    meters = _build_meters(stage_config)

    if loggers:
        logger_hook = None
        for hook in hooks:
            if hasattr(hook, "add_logger"):
                logger_hook = hook
                break

        if logger_hook is None:
            from .hooks.logger import LoggerHook

            logger_hook = LoggerHook()
            hooks.append(logger_hook)

        for logger in loggers:
            logger_hook.add_logger(logger)

    ddp_config = stage_config.get("ddp", {})
    if ddp_config.get("enabled", False):
        from .hooks.ddp import DDPHook

        ddp_hook = DDPHook(
            num_gpus=ddp_config.get("num_gpus", 1),
            find_unused_parameters=ddp_config.get("find_unused_parameters", False),
        )
        hooks.insert(0, ddp_hook)

    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=dataloaders,
        hooks=hooks,
        meters=meters,
    )


def build_loggers(logger_configs: list) -> list:
    """构建 loggers"""
    loggers = []
    for logger_config in logger_configs:
        if isinstance(logger_config, str):
            logger_type = logger_config
            logger_kwargs = {}
        else:
            logger_type = logger_config["type"]
            logger_kwargs = logger_config.get("kwargs", {})

        logger_class = import_module_str(logger_type)
        loggers.append(logger_class(**logger_kwargs))
    return loggers


def _build_model(stage_config: Dict[str, Any]):
    """构建模型"""
    model_type = stage_config["model"]
    model_args = stage_config.get("model_args", {})

    model_class = import_module_str(model_type)
    return model_class(**model_args)


def _build_criterion(stage_config: Dict[str, Any]):
    """构建损失函数"""
    criterion_type = stage_config["criterion"]
    criterion_args = stage_config.get("criterion_args", {})

    criterion_class = import_module_str(criterion_type)
    return criterion_class(**criterion_args)


def _build_optimizer(stage_config: Dict[str, Any], model):
    """构建优化器"""
    from .optim import build_optimizer

    optimizer_config = stage_config["optimizer"]
    return build_optimizer(model, optimizer_config)


def _build_dataloaders(stage_config: Dict[str, Any]):
    """构建数据加载器"""
    from .data.collate import default_collate_fn
    from .data.transforms.utils import parse_transforms_config
    from .data.tensorizer import Tensorizer
    from torch.utils.data import DataLoader

    dataloaders = {}

    for split in ["train", "val"]:
        if split not in stage_config.get("dataloader", {}):
            continue

        loader_config = stage_config["dataloader"][split]

        dataset_config = loader_config.get("dataset", {})
        transforms_config = loader_config.get("transforms", {})
        tensorizer_config = loader_config.get("tensorizer", {})
        loader_args = loader_config.get("dataloader_args", {})

        transforms = parse_transforms_config(transforms_config)

        tensorizer_mapping = {}
        for field_name, field_config in tensorizer_config.get("mapping", {}).items():
            field_type = field_config["type"]
            field_kwargs = field_config.get("kwargs", {})
            field_class = import_module_str(field_type)
            tensorizer_mapping[field_name] = field_class(**field_kwargs)

        tensorizer = Tensorizer(tensorizer_mapping)

        dataset_type = dataset_config.get("type", "cvtrainer.data.BaseDataAdapter")
        dataset_args = dataset_config.get("args", {})

        dataset_class = import_module_str(dataset_type)
        dataset = dataset_class(
            data_path=dataset_config.get("data_path", dataset_args.get("data_path", "")),
            transforms=transforms,
            tensorizer=tensorizer,
            **{k: v for k, v in dataset_args.items() if k != "data_path"},
        )

        collate_fn_name = loader_args.get("collate_fn", "default_collate_fn")
        if collate_fn_name == "default_collate_fn":
            collate_fn = default_collate_fn
        else:
            collate_fn = import_module_str(collate_fn_name)

        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            **{
                k: v
                for k, v in loader_args.items()
                if k not in ["collate_fn", "dataset", "transforms", "tensorizer"]
            },
        )

        dataloaders[split] = dataloader

    return dataloaders


def _build_hooks(stage_config: Dict[str, Any]):
    """构建 Hooks"""
    hooks = []
    hooks_config = stage_config.get("hooks", [])

    for hook_config in hooks_config:
        if isinstance(hook_config, str):
            hook_type = hook_config
            hook_args = {}
        else:
            hook_type = hook_config["type"]
            hook_args = hook_config.get("kwargs", {})

        hook_class = import_module_str(hook_type)
        hooks.append(hook_class(**hook_args))

    return hooks


def _build_meters(stage_config: Dict[str, Any]):
    """构建 Meters"""
    from .meters import (
        LossMeter,
        AccuracyMeter,
        F1Meter,
        PrecisionMeter,
        RecallMeter,
        SegmentationIoUMeter,
        DetectionMapMeter,
    )

    meters = {"loss": LossMeter()}
    meters_config = stage_config.get("meters", [])

    for meter_config in meters_config:
        if isinstance(meter_config, str):
            meter_map = {
                "accuracy": (AccuracyMeter, {}),
                "f1": (F1Meter, {"num_classes": 10}),
                "precision": (PrecisionMeter, {"num_classes": 10}),
                "recall": (RecallMeter, {"num_classes": 10}),
                "miou": (SegmentationIoUMeter, {"num_classes": 21}),
                "map": (DetectionMapMeter, {"num_classes": 80}),
            }

            meter_name = meter_config.lower()
            if meter_name in meter_map:
                meter_class, default_kwargs = meter_map[meter_name]
                meters[meter_name] = meter_class(**default_kwargs)
        else:
            meter_type = meter_config["type"]
            meter_args = meter_config.get("kwargs", {})
            meter_class = import_module_str(meter_type)
            meters[meter_config.get("name", meter_type)] = meter_class(**meter_args)

    return meters
