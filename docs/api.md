# CVTrainer API 参考

## 配置 API

### load_yaml(content, file_path=None)

加载 YAML 配置字符串。

**参数**：
- `content` (str): YAML 内容
- `file_path` (Path, optional): 文件路径（用于解析相对路径）

**返回**：
- `Dict`: 解析后的配置字典

**示例**：
```python
from cvtrainer.core import load_yaml

yaml_str = """
test:
  backend: "Albumentations"
  template: "!include ./common/transforms.yaml::aug_medium"
"""

config = load_yaml(yaml_str)
```

### load_yaml_file(file_path)

从文件加载 YAML 配置。

**参数**：
- `file_path` (str): 配置文件路径

**返回**：
- `Dict`: 解析后的配置字典

**示例**：
```python
from cvtrainer.core import load_yaml_file

config = load_yaml_file("configs/apps/food_classifier/train.yaml")
```

## 模型 API

### BaseModel

三段式模型基类。

**初始化**：
```python
BaseModel(backbone, adapter, head)
```

**参数**：
- `backbone` (BaseBackbone): Backbone 实例
- `adapter` (BaseAdapter): Adapter 实例
- `head` (BaseHead): Head 实例

**方法**：
```python
forward(x) -> torch.Tensor
```

**示例**：
```python
from cvtrainer.models.base import BaseModel
from cvtrainer.models.backbones import ResNet18Backbone
from cvtrainer.models.adapters import ConvAdapter
from cvtrainer.models.heads import MulticlassClassifier

backbone = ResNet18Backbone(pretrained=True)
adapter = ConvAdapter(backbone.backbone_channel, head_channel=256)
head = MulticlassClassifier(adapter.head_channel, num_classes=101)

model = BaseModel(backbone, adapter, head)

x = torch.randn(2, 3, 224, 224)
output = model(x)
```

### BaseBackbone

Backbone 基类。

**属性**：
```python
backbone_channel: int
```

**返回**：
- `int`: Backbone 输出通道数

**方法**：
```python
forward(x) -> torch.Tensor
```

### ResNet18Backbone

ResNet18 Backbone。

**初始化**：
```python
ResNet18Backbone(pretrained=True)
```

**参数**：
- `pretrained` (bool): 是否使用 ImageNet 预训练权重

**属性**：
- `backbone_channel`: 512

### ResNet50Backbone

ResNet50 Backbone。

**初始化**：
```python
ResNet50Backbone(pretrained=True)
```

**参数**：
- `pretrained` (bool): 是否使用 ImageNet 预训练权重

**属性**：
- `backbone_channel`: 2048

### BaseAdapter

Adapter 基类。

**属性**：
```python
head_channel: int
```

**返回**：
- `int`: Adapter 输出通道数

**方法**：
```python
forward(x) -> torch.Tensor
```

### IdentityAdapter

Identity Adapter，不改变通道数。

**初始化**：
```python
IdentityAdapter(backbone_channel)
```

**参数**：
- `backbone_channel` (int): 输入通道数

**属性**：
- `head_channel`: backbone_channel

### ConvAdapter

Conv Adapter: Conv1x1 + BN + ReLU。

**初始化**：
```python
ConvAdapter(backbone_channel, head_channel)
```

**参数**：
- `backbone_channel` (int): 输入通道数
- `head_channel` (int): 输出通道数

### LinearAdapter

Linear Adapter: Global Average Pooling + Linear。

**初始化**：
```python
LinearAdapter(backbone_channel, head_channel)
```

**参数**：
- `backbone_channel` (int): 输入通道数
- `head_channel` (int): 输出通道数

### BaseHead

Head 基类。

**方法**：
```python
forward(x) -> torch.Tensor
```

### MulticlassClassifier

多分类 Head。

**初始化**：
```python
MulticlassClassifier(head_channel, num_classes, dropout_rate=0.5)
```

**参数**：
- `head_channel` (int): 输入通道数
- `num_classes` (int): 类别数
- `dropout_rate` (float): Dropout 比例

### MultiLabelClassifier

多标签分类 Head。

**初始化**：
```python
MultiLabelClassifier(head_channel, num_classes, dropout_rate=0.5)
```

**参数**：
- `head_channel` (int): 输入通道数
- `num_classes` (int): 类别数
- `dropout_rate` (float): Dropout 比例

## 数据引擎 API

### BaseDataAdapter

DataAdapter 基类。

**初始化**：
```python
BaseDataAdapter(data_path, transforms=None, tensorizer=None)
```

**参数**：
- `data_path` (str): 数据路径
- `transforms` (BaseTransform, optional): Transforms 实例
- `tensorizer` (Tensorizer, optional): Tensorizer 实例

**方法**：
```python
__len__() -> int
__getitem__(idx) -> Dict[str, Any]
_load_sample(idx) -> Dict[str, Any]
```

### ImageFolderDataAdapter

ImageFolder DataAdapter。

**初始化**：
```python
ImageFolderDataAdapter(data_path, transforms=None, tensorizer=None)
```

**参数**：
- `data_path` (str): ImageFolder 格式数据路径
- `transforms` (BaseTransform, optional): Transforms 实例
- `tensorizer` (Tensorizer, optional): Tensorizer 实例

**属性**：
- `classes` (List[str]): 类别名称列表
- `class_to_idx` (Dict[str, int]): 类别到索引的映射

### BaseTransform

Transform 基类。

**方法**：
```python
__call__(image) -> Any
```

### AlbumentationsTransform

Albumentations Transform。

**初始化**：
```python
AlbumentationsTransform(transforms)
```

**参数**：
- `transforms` (list): Albumentations transforms 列表

### TorchvisionTransform

Torchvision Transform。

**初始化**：
```python
TorchvisionTransform(transforms)
```

**参数**：
- `transforms` (list): Torchvision transforms 列表

### parse_transforms_config

解析 transforms 配置。

**参数**：
- `config` (Dict): transforms 配置字典

**返回**：
- `Transform`: Transform 实例

**配置格式**：
```python
{
    "backend": "Albumentations",  # 或 "Torchvision"
    "transforms": [
        {"type": "Resize", "kwargs": {"height": 256, "width": 256}},
        {"type": "HorizontalFlip", "kwargs": {"p": 0.5}}
    ]
}
```

### BaseTensorizer

Tensorizer 基类。

**方法**：
```python
__call__(value) -> torch.Tensor
```

### Tensorizer

Composite Tensorizer。

**初始化**：
```python
Tensorizer(field_mapping)
```

**参数**：
- `field_mapping` (Dict[str, BaseTensorizer]): 字段到 Tensorizer 的映射

**方法**：
```python
__call__(sample) -> Dict[str, torch.Tensor]
```

### ImageTensorizer

图像 Tensorizer。

**初始化**：
```python
ImageTensorizer(normalize="imagenet", resize=None, dtype="float")
```

**参数**：
- `normalize` (str): 归一化方式（"imagenet" 或自定义）
- `resize` (tuple, optional): (height, width)
- `dtype` (str): "float" 或 "long"

### OneHotTensorizer

One-Hot / Multi-Hot Tensorizer。

**初始化**：
```python
OneHotTensorizer(num_classes, dtype="float")
```

**参数**：
- `num_classes` (int): 类别数
- `dtype` (str): "float" 或 "long"

### LongTensorizer

Long Tensorizer。

**初始化**：
```python
LongTensorizer()
```

### FloatTensorizer

Float Tensorizer。

**初始化**：
```python
FloatTensorizer()
```

### MaskTensorizer

Mask Tensorizer。

**初始化**：
```python
MaskTensorizer(dtype="long")
```

**参数**：
- `dtype` (str): "float" 或 "long"

### BBoxTensorizer

BBox Tensorizer。

**初始化**：
```python
BBoxTensorizer()
```

### LabelsTensorizer

Labels Tensorizer。

**初始化**：
```python
LabelsTensorizer()
```

### smart_collate_fn

智能 collate 函数，自动处理可变长度。

**参数**：
- `batch` (List[Dict]): batch 数据

**返回**：
- `Dict[str, torch.Tensor]`: batched tensor 字典

### default_collate_fn

默认 collate 函数，只处理固定长度。

**参数**：
- `batch` (List[Dict]): batch 数据

**返回**：
- `Dict[str, torch.Tensor]`: batched tensor 字典

## Hook API

### BaseHook

Hook 基类，定义 16 个调用点。

**方法**：
```python
before_train(stage)
after_train(stage)
before_epoch(stage)
after_epoch(stage)
before_step(stage)
after_step(stage)
before_forward(stage)
after_forward(stage)
before_backward(stage)
after_backward(stage)
before_optimize(stage)
after_optimize(stage)
before_eval(stage)
after_eval(stage)
before_save(stage)
after_save(stage)
```

### LRSchedulerHook

学习率调度 Hook。

**初始化**：
```python
LRSchedulerHook(scheduler_type, scheduler_kwargs)
```

**参数**：
- `scheduler_type` (str): Scheduler 类型（如 "CosineAnnealingLR"）
- `scheduler_kwargs` (Dict): Scheduler 参数

**方法**：
```python
setup_scheduler(optimizer)
after_epoch(stage)
```

### EMAHook

指数移动平均 Hook。

**初始化**：
```python
EMAHook(decay=0.999)
```

**参数**：
- `decay` (float): EMA 衰减率

**方法**：
```python
before_train(stage)
after_step(stage)
```

### CheckpointHook

检查点保存 Hook。

**初始化**：
```python
CheckpointHook(save_dir="checkpoints", save_freq=1)
```

**参数**：
- `save_dir` (str): 保存目录
- `save_freq` (int): 保存频率（epoch 间隔）

**方法**：
```python
before_train(stage)
after_epoch(stage)
_save_checkpoint(stage)
```

## Meter API

### BaseMeter

Meter 基类。

**方法**：
```python
update(value)
get_value() -> Any
reset()
```

### LossMeter

Loss Meter。

**初始化**：
```python
LossMeter()
```

**方法**：
```python
update(value: float)
get_value() -> float
reset()
```

### AccuracyMeter

准确率 Meter。

**初始化**：
```python
AccuracyMeter(topk=(1,))
```

**参数**：
- `topk` (tuple): top-k 值列表

**方法**：
```python
update(output, target)
get_value() -> List[float]
reset()
```

## 优化器 API

### build_optimizer

构建优化器，支持分层学习率。

**参数**：
- `model` (nn.Module): 模型
- `config` (Dict): 优化器配置

**返回**：
- `Optimizer`: Optimizer 实例

**配置格式**：
```python
{
    "type": "Adam",
    "kwargs": {"lr": 0.001},
    "layerwise": {
        "backbone": 0.001,
        "adapter": 0.01,
        "head": 0.1
    }
}
```

### build_scheduler

构建学习率调度器。

**参数**：
- `optimizer` (Optimizer): Optimizer 实例
- `config` (Dict): 调度器配置

**返回**：
- `LRScheduler`: Scheduler 实例

**配置格式**：
```python
{
    "type": "CosineAnnealingLR",
    "kwargs": {"T_max": 100}
}
```

## DDP API

### setup_ddp

设置 DDP。

```python
setup_ddp()
```

### cleanup_ddp

清理 DDP。

```python
cleanup_ddp()
```

### is_main_process

判断是否为主进程。

**返回**：
- `bool`: 是否为主进程

### get_world_size

获取 world size。

**返回**：
- `int`: GPU 数量

### get_rank

获取 rank。

**返回**：
- `int`: 当前进程 rank

### launch_ddp

启动 DDP 训练。

**参数**：
- `train_fn`: 训练函数
- `world_size`: GPU 数量
- `*args`: 传递给 train_fn 的位置参数
- `**kwargs`: 传递给 train_fn 的关键字参数

## CLI API

### main

CLI 入口函数。

**使用**：
```bash
cvtrainer --config <config_file>
```

**参数**：
- `--config` (str): 配置文件路径（必需）

## Builder API

### build_stage

构建 Stage 实例。

**参数**：
- `stage_config` (Dict): Stage 配置字典

**返回**：
- `Stage`: Stage 实例

### import_module_str

通过 importlib 导入模块。

**参数**：
- `module_str` (str): 模块字符串

**返回**：
- `Any`: 类或函数

**示例**：
```python
from cvtrainer.builder import import_module_str

model_class = import_module_str("cvtrainer.models.backbones.ResNet18Backbone")
model = model_class(pretrained=True)
```
