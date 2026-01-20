# CVTrainer 最佳实践

## 配置管理

### 1. 使用配置模板

创建通用配置模板，避免重复：

```yaml
# configs/common/transforms.yaml
aug_light:
  backend: "Albumentations"
  transforms:
    - type: "Resize"
      kwargs: {height: 256, width: 256}
    - type: "CenterCrop"
      kwargs: {height: 224, width: 224}

aug_medium:
  backend: "Albumentations"
  transforms:
    - type: "Resize"
      kwargs: {height: 256, width: 256}
    - type: "RandomResizedCrop"
      kwargs: {height: 224, width: 224}
    - type: "HorizontalFlip"
      kwargs: {p: 0.5}

aug_heavy:
  backend: "Albumentations"
  transforms:
    - type: "Resize"
      kwargs: {height: 256, width: 256}
    - type: "RandomResizedCrop"
      kwargs: {height: 224, width: 224}
    - type: "HorizontalFlip"
      kwargs: {p: 0.5}
    - type: "VerticalFlip"
      kwargs: {p: 0.5}
    - type: "RandomRotate90"
      kwargs: {p: 0.5}
```

### 2. 配置文件组织

推荐的项目配置结构：

```
configs/
├── common/
│   ├── transforms.yaml       # 通用数据增强
│   ├── tensorizers.yaml      # 通用 tensorizer
│   ├── optimizers.yaml       # 通用优化器配置
│   └── schedulers.yaml       # 通用学习率调度
├── apps/
│   └── food_classifier/
│       ├── train.yaml         # 训练配置
│       ├── eval.yaml          # 评估配置
│       └── debug.yaml         # 调试配置
```

### 3. 模板复写最佳实践

**场景 1：只修改少量参数**

```yaml
transforms:
  template: "!include ../../common/transforms.yaml::aug_medium"
  backend: "Torchvision"  # 只改 backend
```

**场景 2：扩展 transforms**

```yaml
transforms:
  template: "!include ../../common/transforms.yaml::aug_medium"
  transforms:
    - type: "Resize"
      kwargs: {height: 256, width: 256}
    - type: "RandomResizedCrop"
      kwargs: {height: 224, width: 224}
    - type: "HorizontalFlip"
      kwargs: {p: 0.5}
    - type: "ColorJitter"  # 新增
      kwargs: {brightness: 0.2, contrast: 0.2}
```

**场景 3：多层嵌套模板**

```yaml
# configs/common/tensorizers.yaml
imagenet_x:
  type: "cvtrainer.data.tensorizer.ImageTensorizer"
  kwargs: {normalize: imagenet, dtype: float}

single_label:
  type: "cvtrainer.data.tensorizer.OneHotTensorizer"
  kwargs: {dtype: float}

# configs/apps/food_classifier/train.yaml
tensorizer:
  type: "cvtrainer.data.tensorizer.Tensorizer"
  mapping:
    x:
      template: "!include ../../../common/tensorizers.yaml::imagenet_x"
    class_id:
      template: "!include ../../../common/tensorizers.yaml::single_label"
      kwargs: {num_classes: 101}  # 复写 num_classes
```

## 模型设计

### 1. 三段式模型最佳实践

**选择合适的 Backbone**：

```python
# 小模型/端侧：MobileNet
from cvtrainer.models.backbones import MobileNetV2Backbone

# 中等模型：ResNet18
from cvtrainer.models.backbones import ResNet18Backbone

# 大模型：ResNet50
from cvtrainer.models.backbones import ResNet50Backbone
```

**选择合适的 Adapter**：

```python
# 无需降维：Identity
from cvtrainer.models.adapters import IdentityAdapter

# 轻微降维：Conv
from cvtrainer.models.adapters import ConvAdapter
adapter = ConvAdapter(backbone_channel=512, head_channel=256)

# 大幅降维：Linear
from cvtrainer.models.adapters import LinearAdapter
adapter = LinearAdapter(backbone_channel=2048, head_channel=256)
```

**选择合适的 Head**：

```python
# 多分类
from cvtrainer.models.heads import MulticlassClassifier
head = MulticlassClassifier(head_channel=256, num_classes=101)

# 多标签
from cvtrainer.models.heads import MultiLabelClassifier
head = MultiLabelClassifier(head_channel=256, num_classes=101)
```

### 2. 自定义模型示例

```python
# apps/my_project/model.py
from cvtrainer.models.base import BaseModel
from cvtrainer.models.backbones import ResNet18Backbone
from cvtrainer.models.adapters import ConvAdapter
from cvtrainer.models.heads import MulticlassClassifier

class MyModel(BaseModel):
    """
    自定义模型
    
    Args:
        num_classes (int): 类别数
        dropout_rate (float): Dropout 比例
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        # Backbone
        backbone = ResNet18Backbone(pretrained=True)
        
        # Adapter（降维到 256）
        adapter = ConvAdapter(
            backbone_channel=backbone.backbone_channel,
            head_channel=256
        )
        
        # Head
        head = MulticlassClassifier(
            head_channel=adapter.head_channel,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        super().__init__(backbone, adapter, head)
```

## 数据处理

### 1. 数据增强策略

**训练阶段**：使用强增强

```yaml
transforms:
  backend: "Albumentations"
  transforms:
    - type: "Resize"
      kwargs: {height: 256, width: 256}
    - type: "RandomResizedCrop"
      kwargs: {height: 224, width: 224}
    - type: "HorizontalFlip"
      kwargs: {p: 0.5}
    - type: "ColorJitter"
      kwargs: {brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.1}
    - type: "ToGray"
      kwargs: {p: 0.1}
```

**验证阶段**：使用弱增强或无增强

```yaml
transforms:
  backend: "Albumentations"
  transforms:
    - type: "Resize"
      kwargs: {height: 256, width: 256}
    - type: "CenterCrop"
      kwargs: {height: 224, width: 224}
```

### 2. Tensorizer 配置最佳实践

**图像字段**：使用 ImageTensorizer

```yaml
x:
  type: "cvtrainer.data.tensorizer.ImageTensorizer"
  kwargs:
    normalize: imagenet  # ImageNet 标准化
    dtype: float
```

**分类标签**：使用 OneHotTensorizer

```yaml
# 单标签
class_id:
  type: "cvtrainer.data.tensorizer.OneHotTensorizer"
  kwargs:
    num_classes: 101
    dtype: float

# 多标签
labels:
  type: "cvtrainer.data.tensorizer.OneHotTensorizer"
  kwargs:
    num_classes: 101
    dtype: float
```

**检测框**：使用 BBoxTensorizer

```yaml
bboxes:
  type: "cvtrainer.data.tensorizer.BBoxTensorizer"
```

**检测标签**：使用 LabelsTensorizer

```yaml
labels:
  type: "cvtrainer.data.tensorizer.LabelsTensorizer"
```

### 3. Collate 选择

**固定长度数据**：使用 default_collate_fn

```yaml
dataloader_args:
  collate_fn: "cvtrainer.data.collate.default_collate_fn"
```

**可变长度数据**：使用 smart_collate_fn

```yaml
dataloader_args:
  collate_fn: "cvtrainer.data.collate.smart_collate_fn"
```

## 训练技巧

### 1. 分层学习率

**场景**：微调预训练模型

```yaml
optimizer:
  type: "Adam"
  kwargs: {lr: 0.001}
  layerwise:
    backbone: 0.0001  # Backbone 学习率更小
    adapter: 0.001    # Adapter 正常
    head: 0.01       # Head 学习率更大
```

### 2. 学习率调度

**场景 1**：Cosine Annealing（推荐）

```yaml
hooks:
  - type: "cvtrainer.hooks.LRSchedulerHook"
    kwargs:
      scheduler_type: "CosineAnnealingLR"
      scheduler_kwargs: {T_max: 100}
```

**场景 2**：Step Decay

```yaml
hooks:
  - type: "cvtrainer.hooks.LRSchedulerHook"
    kwargs:
      scheduler_type: "StepLR"
      scheduler_kwargs: {step_size: 30, gamma: 0.1}
```

**场景 3**：Multi-Step Decay

```yaml
hooks:
  - type: "cvtrainer.hooks.LRSchedulerHook"
    kwargs:
      scheduler_type: "MultiStepLR"
      scheduler_kwargs: {milestones: [50, 80], gamma: 0.1}
```

### 3. EMA 使用

**训练阶段**：启用 EMA

```yaml
hooks:
  - type: "cvtrainer.hooks.EMAHook"
    kwargs:
      decay: 0.999
```

**评估阶段**：使用 EMA 模型

```python
# 在评估时使用 EMA 模型
from cvtrainer.hooks import EMAHook

# 获取 EMA 模型
ema_hook = next(h for h in hooks if isinstance(h, EMAHook))
ema_model = ema_hook.ema_model

# 使用 EMA 模型评估
outputs = ema_model(inputs)
```

### 4. 检查点管理

**定期保存**：

```yaml
hooks:
  - type: "cvtrainer.hooks.CheckpointHook"
    kwargs:
      save_dir: "checkpoints"
      save_freq: 10  # 每 10 个 epoch 保存一次
```

**保存最佳模型**：自定义 Hook

```python
class BestModelHook(BaseHook):
    def __init__(self, save_dir, metric_key="accuracy"):
        self.save_dir = save_dir
        self.metric_key = metric_key
        self.best_metric = 0.0
    
    def after_epoch(self, stage):
        metric = stage["meters"][self.metric_key].get_value()
        
        if metric > self.best_metric:
            self.best_metric = metric
            
            checkpoint = {
                "epoch": stage["epoch"],
                "model_state_dict": stage["model"].state_dict(),
                "best_metric": self.best_metric,
            }
            
            torch.save(checkpoint, f"{self.save_dir}/best_model.pth")
```

## 性能优化

### 1. DataLoader 优化

```yaml
dataloader_args:
  batch_size: 32
  num_workers: 4        # 多进程加载
  pin_memory: true      # 加速 GPU 传输
  prefetch_factor: 2    # 预取批次
  persistent_workers: true  # 保持 worker 进程
```

### 2. 混合精度训练

自定义 Hook 实现 AMP：

```python
class AMPHook(BaseHook):
    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()
    
    def before_forward(self, stage):
        stage["use_amp"] = True
    
    def after_forward(self, stage):
        if stage.get("use_amp"):
            stage["loss"] = self.scaler.scale(stage["loss"])
    
    def after_backward(self, stage):
        if stage.get("use_amp"):
            self.scaler.unscale_(stage["optimizer"])
            torch.nn.utils.clip_grad_norm_(stage["model"].parameters(), 1.0)
            self.scaler.step(stage["optimizer"])
            self.scaler.update()
```

### 3. 梯度累积

```yaml
dataloader_args:
  batch_size: 8   # 单张 GPU 批次

hooks:
  - type: "GradientAccumulationHook"
    kwargs:
      accumulation_steps: 4  # 等效 batch_size = 32
```

## 调试技巧

### 1. 使用小数据集调试

```yaml
dataloader:
  train:
    dataset:
      data_path: "debug_data"  # 小数据集
    dataloader_args:
      batch_size: 4  # 小批次
```

### 2. 打印配置

```python
from cvtrainer.core import load_yaml_file

config = load_yaml_file("configs/apps/food_classifier/train.yaml")
import json
print(json.dumps(config, indent=2))
```

### 3. 验证配置

```python
from cvtrainer.builder import build_stage

stage_config = load_yaml_file("configs/apps/food_classifier/train.yaml")["stages"][0]
stage = build_stage(stage_config)

# 检查组件
print(f"Model: {type(stage.model).__name__}")
print(f"Criterion: {type(stage.criterion).__name__}")
print(f"Optimizer: {type(stage.optimizer).__name__}")
print(f"Dataloaders: {list(stage.dataloaders.keys())}")
print(f"Hooks: {[type(h).__name__ for h in stage.hooks]}")
print(f"Meters: {list(stage.meters.keys())}")
```

## 常见问题解决

### 1. CUDA OOM

**解决方案**：
- 减小 batch_size
- 减小模型（使用更小的 backbone）
- 使用梯度累积

```yaml
dataloader_args:
  batch_size: 16  # 减小 batch_size

optimizer:
  type: "Adam"
  kwargs: {lr: 0.001}
  layerwise:
    backbone: 0.0001
    adapter: 0.001
    head: 0.01

# 或使用梯度累积
hooks:
  - type: "GradientAccumulationHook"
    kwargs:
      accumulation_steps: 4
```

### 2. 过拟合

**解决方案**：
- 增加数据增强
- 使用 Dropout
- 使用 EMA
- 早停（自定义 Hook）

```yaml
# 增加数据增强
transforms:
  backend: "Albumentations"
  transforms:
    - type: "RandomResizedCrop"
      kwargs: {height: 224, width: 224}
    - type: "HorizontalFlip"
      kwargs: {p: 0.5}
    - type: "ColorJitter"
      kwargs: {brightness: 0.2, contrast: 0.2}

# 使用 Dropout
model_args:
  dropout_rate: 0.5

# 使用 EMA
hooks:
  - type: "cvtrainer.hooks.EMAHook"
    kwargs: {decay: 0.999}
```

### 3. 学习率不下降

**检查**：
- Scheduler Hook 是否正确配置
- Scheduler 是否正确更新

```yaml
hooks:
  - type: "cvtrainer.hooks.LRSchedulerHook"
    kwargs:
      scheduler_type: "CosineAnnealingLR"
      scheduler_kwargs: {T_max: 100}
```

### 4. 配置文件引用错误

**错误示例**：
```yaml
# 相对路径错误
transforms: "!include transforms.yaml"  # 错误：相对路径错误
```

**正确示例**：
```yaml
# 使用相对路径
transforms: "!include ../../common/transforms.yaml::aug_medium"

# 或使用绝对路径
transforms: "!include /path/to/transforms.yaml::aug_medium"
```

## 迁移指南

### 从其他框架迁移

**MMSegmentation → CVTrainer**：

```python
# MMSegmentation 模型
class MySegmentor(nn.Module):
    def __init__(self, backbone, decode_head):
        self.backbone = backbone
        self.decode_head = decode_head

# CVTrainer 模型
class MyModel(BaseModel):
    def __init__(self, num_classes):
        backbone = ResNet18Backbone(pretrained=True)
        adapter = IdentityAdapter(backbone.backbone_channel)
        head = SegmentationHead(adapter.head_channel, num_classes)
        super().__init__(backbone, adapter, head)
```

**PyTorch Lightning → CVTrainer**：

```yaml
# PyTorch Lightning
model = LitModel()

# CVTrainer
model: "apps.my_project.model.MyModel"
model_args: {num_classes: 101}
```

## 部署建议

### 1. 模型导出

导出 ONNX 模型：

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
```

### 2. 模型量化

使用 PyTorch 量化：

```python
import torch.quantization

model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 3. 模型优化

使用 TorchScript：

```python
model.eval()
scripted_model = torch.jit.script(model)

torch.jit.save(scripted_model, "model.pt")
```
