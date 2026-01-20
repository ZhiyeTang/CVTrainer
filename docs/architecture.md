# CVTrainer 架构设计

## 概述

CVTrainer 是一个配置驱动的端侧视觉任务训练框架，采用模块化设计理念，将训练流程分解为可配置的组件。

## 设计原则

### 1. 配置驱动

所有训练逻辑通过 YAML 配置文件定义，无需编写训练代码：

```yaml
stages:
  - name: "train"
    model: "apps.my_project.model.MyModel"
    model_args: {num_classes: 101}
```

### 2. 模块化

每个组件独立开发、测试和配置：

- Models: Backbone → Adapter → Head
- Data: Dataset → Transforms → Tensorizer → Collate
- Training: Criterion → Optimizer → Hooks → Meters

### 3. 灵活性

通过 Python import 语法加载任意模块：

```yaml
model: "apps.my_project.model.MyModel"  # 自定义模型
criterion: "torch.nn.CrossEntropyLoss"   # PyTorch 内置
loss: "cvtrainer.loss.BCELoss"          # 框架内置
```

## 核心架构

### 整体流程

```
YAML Config
    ↓
Config Parser (!include, template)
    ↓
Builder
    ↓
Stage {model, criterion, optimizer, dataloaders, hooks, meters}
    ↓
Trainer
    ↓
Training Loop (with Hooks)
```

### 组件关系图

```
┌─────────────────────────────────────────────────────────────┐
│                         YAML Config                         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      Config Parser                          │
│  - !include: 外部文件引用                                    │
│  - template: 模板复写扩展                                    │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                        Builder                              │
│  - build_model()                                            │
│  - build_criterion()                                        │
│  - build_optimizer()                                        │
│  - build_dataloaders()                                      │
│  - build_hooks()                                            │
│  - build_meters()                                           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                         Stage                               │
│  - model: BaseModel                                         │
│  - criterion: nn.Module                                     │
│  - optimizer: Optimizer                                     │
│  - dataloaders: Dict[str, DataLoader]                       │
│  - hooks: List[BaseHook]                                    │
│  - meters: Dict[str, BaseMeter]                             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                        Trainer                              │
│  - train()                                                  │
│  - evaluate()                                               │
│  - save_checkpoint()                                        │
│  - load_checkpoint()                                        │
└─────────────────────────────────────────────────────────────┘
```

## 模块详解

### 1. 配置系统

#### 设计目标

- 支持外部配置复用
- 支持模板继承和复写
- 配置即文档，简洁易读

#### 核心组件

**ConfigLoader**: 扩展 `yaml.SafeLoader`，支持自定义标签

```python
class ConfigLoader(yaml.SafeLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loading_files: set = set()
```

**!include 构造器**: 处理外部文件引用

```python
def include_constructor(loader, node):
    value = loader.construct_scalar(node)
    
    # 解析文件路径和 key
    if "::" in value:
        file_path, key = value.split("::", 1)
    else:
        file_path, key = value, None
    
    # 解析相对路径
    current_dir = Path(loader.stream.name).parent
    full_path = (current_dir / file_path).resolve()
    
    # 递归加载
    with open(full_path, "r") as f:
        data = load_yaml(f.read(), full_path)
    
    return data[key] if key else data
```

**template 字段解析**: 递归深度合并

```python
def resolve_template_field(config):
    if not isinstance(config, dict):
        return config
    
    result = config.copy()
    
    if "template" in result:
        template_config = result.pop("template")
        
        if isinstance(template_config, dict):
            template_config = resolve_template_field(template_config)
            result = deep_update(template_config, result)
        else:
            result = template_config
    
    for key, value in result.items():
        result[key] = resolve_template_field(value)
    
    return result

def deep_update(target, source):
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            target[key] = deep_update(target[key], value)
        else:
            target[key] = value
    return target
```

#### 配置解析流程

```
1. YAML 解析（!include 标签）
   ↓
2. template 字段解析（递归深度合并）
   ↓
3. 最终配置字典
```

### 2. 模型系统

#### 三段式架构

```
Input Image
    ↓
┌──────────────┐
│  Backbone    │ (特征提取)
│  - ResNet    │
│  - MobileNet │
│  - Custom    │
└──────┬───────┘
       │ backbone_channel
       ↓
┌──────────────┐
│   Adapter    │ (特征转换)
│  - Identity  │
│  - Conv      │
│  - Linear    │
└──────┬───────┘
       │ head_channel
       ↓
┌──────────────┐
│    Head      │ (任务特定)
│  - Classifier│
│  - Detector  │
│  - Segmenter │
└──────┬───────┘
       ↓
   Output
```

#### Base 类设计

**BaseModel**: 组合三段式模型

```python
class BaseModel(nn.Module):
    def __init__(self, backbone, adapter, head):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.adapter(x)
        return self.head(x)
```

**BaseBackbone**: 特征提取基类

```python
class BaseBackbone(nn.Module):
    @property
    @abstractmethod
    def backbone_channel(self) -> int:
        """输出通道数"""
        pass
```

**BaseAdapter**: 特征转换基类

```python
class BaseAdapter(nn.Module):
    @property
    @abstractmethod
    def head_channel(self) -> int:
        """输出通道数"""
        pass
```

**BaseHead**: 任务特定头部基类

```python
class BaseHead(nn.Module):
    @abstractmethod
    def forward(self, x):
        pass
```

#### 通道流转机制

```
backbone.backbone_channel → adapter.__init__(backbone_channel)
                                      ↓
adapter.head_channel     → head.__init__(head_channel)
```

### 3. 数据引擎

#### 数据流程

```
原始文件
    ↓
DataAdapter._load_sample()  → {x: PIL.Image, target: dict}
    ↓
Transforms                 → {x: augmented_image, target: dict}
    ↓
Tensorizer (field mapping)  → {x: tensor, target: tensor_dict}
    ↓
Collate Function           → batch: {x: (B,C,H,W), target: ...}
    ↓
Trainer
```

#### DataAdapter 设计

**职责边界**：
- `_load_sample()`: 读取原始数据，返回 `{x, target}`
- `__getitem__()`: 应用 transforms 和 tensorizer
- 不负责 collate（由 DataLoader 的 collate_fn 处理）

```python
class BaseDataAdapter(Dataset):
    def __init__(self, data_path, transforms=None, tensorizer=None):
        self.data_path = data_path
        self.transforms = transforms
        self.tensorizer = tensorizer
    
    def __getitem__(self, idx):
        sample = self._load_sample(idx)
        
        if self.transforms:
            sample["x"] = self.transforms(sample["x"])
        
        if self.tensorizer:
            sample = self.tensorizer(sample)
        
        return sample
```

#### Transforms 设计

**Backend 切换**：通过 `backend` 字段指定

```yaml
transforms:
  backend: "Albumentations"  # 或 "Torchvision"
  transforms:
    - type: "Resize"
      kwargs: {height: 256, width: 256}
```

**解析逻辑**：

```python
def parse_transforms_config(config):
    backend_name = config["backend"]
    
    if backend_name == "Albumentations":
        backend_module = "albumentations"
        transform_class = AlbumentationsTransform
    elif backend_name == "Torchvision":
        backend_module = "torchvision.transforms"
        transform_class = TorchvisionTransform
    
    parsed_transforms = []
    for transform_config in config["transforms"]:
        transform = parse_transform_item(transform_config, backend_module)
        parsed_transforms.append(transform)
    
    return transform_class(parsed_transforms)
```

#### Tensorizer 设计

**字段级别配置**：每个字段独立的 Tensorizer

```yaml
tensorizer:
  type: "cvtrainer.data.tensorizer.Tensorizer"
  mapping:
    x:
      type: "cvtrainer.data.tensorizer.ImageTensorizer"
      kwargs: {normalize: imagenet}
    class_id:
      type: "cvtrainer.data.tensorizer.OneHotTensorizer"
      kwargs: {num_classes: 101}
```

**Composite 模式**：

```python
class Tensorizer:
    def __init__(self, field_mapping):
        self.field_mapping = field_mapping
    
    def __call__(self, sample):
        result = {}
        for key, value in sample.items():
            if key not in self.field_mapping:
                raise ValueError(f"Field '{key}' not configured")
            result[key] = self.field_mapping[key](value)
        return result
```

**BaseTensorizer 接口**：

```python
class BaseTensorizer(ABC):
    @abstractmethod
    def __call__(self, value) -> torch.Tensor:
        pass
```

#### Collate 设计

**智能 Collate**：自动处理可变长度

```python
def smart_collate_fn(batch):
    result = {}
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch]
        first_value = values[0]
        
        if first_value.dim() == 0:
            # 标量 → stack
            result[key] = torch.stack(values)
        elif first_value.dim() == 1:
            # 向量 → pad_sequence
            result[key] = torch.nn.utils.rnn.pad_sequence(
                values, batch_first=True, padding_value=0
            )
        elif first_value.dim() >= 2:
            # 多维 → padding
            result[key] = pad_tensor_sequence(values, padding_value=0)
    
    return result
```

### 4. Hook 系统

#### 调用时机

```
Training Loop:
  before_train()
  for epoch in range(epochs):
    before_epoch()
    for step in range(steps):
      before_step()
      before_forward()
      output = model(x)
      after_forward()
      before_backward()
      loss.backward()
      after_backward()
      before_optimize()
      optimizer.step()
      after_optimize()
      after_step()
    after_epoch()
  after_train()

Evaluation Loop:
  before_eval()
  ...
  after_eval()

Checkpoint Save:
  before_save()
  save_checkpoint()
  after_save()
```

#### BaseHook 接口

```python
class BaseHook(ABC):
    def before_train(self, stage): pass
    def after_train(self, stage): pass
    def before_epoch(self, stage): pass
    def after_epoch(self, stage): pass
    def before_step(self, stage): pass
    def after_step(self, stage): pass
    def before_forward(self, stage): pass
    def after_forward(self, stage): pass
    def before_backward(self, stage): pass
    def after_backward(self, stage): pass
    def before_optimize(self, stage): pass
    def after_optimize(self, stage): pass
    def before_eval(self, stage): pass
    def after_eval(self, stage): pass
    def before_save(self, stage): pass
    def after_save(self, stage): pass
```

#### SchedulerHook 设计

**Scheduler 管理**：Hook 内部构建和管理 scheduler

```python
class LRSchedulerHook(BaseHook):
    def __init__(self, scheduler_type, scheduler_kwargs):
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler = None
    
    def setup_scheduler(self, optimizer):
        scheduler_module = importlib.import_module("torch.optim.lr_scheduler")
        scheduler_class = getattr(scheduler_module, self.scheduler_type)
        self.scheduler = scheduler_class(optimizer, **self.scheduler_kwargs)
    
    def after_epoch(self, stage):
        if self.scheduler:
            self.scheduler.step()
```

#### EMAHook 设计

**EMA 模型维护**：独立的 EMA 模型副本

```python
class EMAHook(BaseHook):
    def __init__(self, decay=0.999):
        self.decay = decay
        self.ema_model = None
        self.shadow = {}
    
    def before_train(self, stage):
        model = stage["model"]
        self.ema_model = type(model)(
            model.backbone,
            model.adapter,
            model.head
        )
        self.ema_model.load_state_dict(model.state_dict())
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def after_step(self, stage):
        model = stage["model"]
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
```

### 5. 优化器系统

#### 分层学习率

**设计思路**：不同模块使用不同学习率

```yaml
optimizer:
  type: "Adam"
  kwargs: {lr: 0.001}
  layerwise:
    backbone: 0.001
    adapter: 0.01
    head: 0.1
```

**参数组构建**：

```python
def build_optimizer(model, config):
    optimizer_type = config["type"]
    kwargs = config.get("kwargs", {})
    layerwise_config = config.get("layerwise", None)
    
    if layerwise_config:
        params = _build_layerwise_params(model, layerwise_config, kwargs["lr"])
    else:
        params = model.parameters()
    
    optimizer_class = getattr(torch.optim, optimizer_type)
    return optimizer_class(params, **kwargs)

def _build_layerwise_params(model, layerwise_config, default_lr):
    params = []
    
    backbone_lr = layerwise_config.get("backbone", default_lr)
    adapter_lr = layerwise_config.get("adapter", default_lr)
    head_lr = layerwise_config.get("head", default_lr)
    
    params.append({"params": model.backbone.parameters(), "lr": backbone_lr})
    params.append({"params": model.adapter.parameters(), "lr": adapter_lr})
    params.append({"params": model.head.parameters(), "lr": head_lr})
    
    return params
```

#### Scheduler 工厂

**通用支持**：支持所有 PyTorch schedulers

```yaml
hooks:
  - type: "cvtrainer.hooks.LRSchedulerHook"
    kwargs:
      scheduler_type: "CosineAnnealingLR"
      scheduler_kwargs: {T_max: 100}
```

```python
def build_scheduler(optimizer, config):
    scheduler_type = config["type"]
    kwargs = config.get("kwargs", {})
    
    scheduler_module = importlib.import_module("torch.optim.lr_scheduler")
    scheduler_class = getattr(scheduler_module, scheduler_type)
    return scheduler_class(optimizer, **kwargs)
```

## 设计决策

### 为什么使用三段式架构？

**优势**：
1. **解耦**：特征提取、转换、任务逻辑分离
2. **复用**：Backbone 和 Adapter 可跨任务复用
3. **灵活性**：不同任务只需替换 Head

**示例**：
```
ResNet18 + Conv + Classifier → 图像分类
ResNet18 + Conv + Detector   → 目标检测
ResNet18 + Conv + Segmenter  → 语义分割
```

### 为什么使用字段级别 Tensorizer？

**优势**：
1. **显式**：每个字段转换逻辑清晰
2. **灵活**：不同字段可使用不同策略
3. **可扩展**：易于添加自定义 Tensorizer

**对比方案**：
- 全局 Tensorizer：无法处理复杂 target 结构
- 自动推断：逻辑不透明，难以调试

### 为什么使用 Hook 而不是 callback？

**优势**：
1. **标准化**：16 个固定调用点，接口一致
2. **可组合**：多个 Hook 可同时工作
3. **解耦**：训练逻辑与优化逻辑分离

**调用流程**：
```
Trainer → _run_hooks("before_train") → 各 hook.before_train()
        → training loop
        → _run_hooks("after_train")  → 各 hook.after_train()
```

### 为什么只支持 kwargs 不支持 args？

**优势**：
1. **可读性**：kwargs 命名清晰
2. **灵活性**：参数顺序无关
3. **YAML 友好**：结构化配置

**示例**：
```yaml
# 好的设计（kwargs）
type: "Resize"
kwargs: {height: 256, width: 256}

# 不推荐（args，顺序依赖）
type: "Resize"
args: [256, 256]  # 哪个是 height 哪个是 width？
```

## 性能优化

### 数据加载优化

1. **多进程 DataLoader**：
   ```yaml
   dataloader_args:
     num_workers: 4
     pin_memory: true
   ```

2. **预取**：
   ```yaml
   dataloader_args:
     prefetch_factor: 2
   ```

### 内存优化

1. **Gradient Accumulation**：通过 batch size 调整
2. **Mixed Precision**：通过 Hook 实现
3. **Gradient Checkpointing**：通过自定义 Backbone 实现

### 计算优化

1. **EMA**：避免频繁的模型拷贝
2. **DDP**：支持多 GPU 并行
3. **Compiler**：未来可集成 torch.compile

## 扩展性

### 添加新 Backbone

```python
class MyBackbone(BaseBackbone):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = my_model(pretrained)
        self._backbone_channel = 512
    
    @property
    def backbone_channel(self):
        return self._backbone_channel
    
    def forward(self, x):
        return self.model(x)
```

### 添加新 Tensorizer

```python
class MyTensorizer(BaseTensorizer):
    def __call__(self, value):
        return torch.tensor(value, dtype=torch.float32)
```

### 添加新 Hook

```python
class MyHook(BaseHook):
    def after_step(self, stage):
        # 自定义逻辑
        pass
```

## 未来规划

1. **更多 Backbones**：EfficientNet, ViT, Swin Transformer
2. **更多任务支持**：检测、分割、关键点检测
3. **自动混合精度**：AMP Hook
4. **分布式优化**：DeepSpeed, FSDP 集成
5. **可视化**：TensorBoard Hook
6. **模型压缩**：剪枝、量化 Hook
