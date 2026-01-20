# CVTrainer

一个配置驱动的端侧视觉任务小模型训练框架。

## 特性

- **配置驱动**：通过 YAML 配置文件定义所有训练阶段
- **灵活的模块系统**：使用 Python import 语法加载模块
- **三段式模型架构**：Backbone → Adapter → Head
- **数据引擎**：DataAdapter、Transforms、Tensorizer、智能 Collate
- **Hook 系统**：16 个调用点，支持自定义训练逻辑
- **极简 Trainer 类**：核心训练基础设施，通过 Hook 扩展所有功能
- **日志系统**：支持 Console、File、TensorBoard、WandB 四种日志方式
- **进度条**：实时显示训练/验证 loss 和指标
- **DDP 支持**：通过 DDPHook 实现多机多卡训练
- **分层学习率**：原生支持差分学习率
- **配置模板**：支持 `!include` 引用和 `template` 复写

## 安装

```bash
# 从源码安装
pip install -e .

# 或安装依赖
pip install -r requirements.txt
```

## 文档

- **[详细文档](./docs/README.md)** - 完整的架构设计、API 参考和最佳实践
- **[架构设计](./docs/architecture.md)** - 框架架构和设计原理
- **[API 参考](./docs/api.md)** - 完整 API 文档
- **[最佳实践](./docs/best_practices.md)** - 使用技巧和优化建议
- **[贡献指南](./docs/contributing.md)** - 如何参与项目开发
- **[更新日志](./docs/changelog.md)** - 版本变更历史

## 快速开始

### 1. 基本使用

```bash
# 使用配置文件训练
cvtrainer train --config apps/cifar10_demo/train.yaml

# 从 checkpoint 恢复训练
cvtrainer train --config apps/cifar10_demo/train.yaml --resume checkpoints/checkpoint_epoch_10.pth

# 评估模型
cvtrainer eval --config apps/cifar10_demo/eval.yaml --checkpoint checkpoints/best_model.pth
```

### 2. DDP 训练

```bash
# 单机多卡
cvtrainer train --config apps/cifar10_demo/train.yaml --num_gpus 4

# 多机多卡需要使用 torchrun
torchrun --nproc_per_node=4 -m cvtrainer.cli train --config apps/cifar10_demo/train.yaml
```

### 3. 配置示例

```yaml
stages:
  - name: "train"
    epochs: 100
    
    model: "apps.my_project.model.MyModel"
    model_args: {num_classes: 101}
    
    criterion: "cvtrainer.loss.CrossEntropyLoss"
    
    optimizer:
      type: "Adam"
      kwargs: {lr: 0.001}
      layerwise:
        backbone: 0.001
        adapter: 0.01
        head: 0.1
    
    dataloader:
      train:
        dataset:
          data_path: "data/train"
        transforms:
          backend: "Albumentations"
          transforms:
            - type: "Resize"
              kwargs: {height: 256, width: 256}
            - type: "RandomResizedCrop"
              kwargs: {height: 224, width: 224}
            - type: "HorizontalFlip"
              kwargs: {p: 0.5}
        tensorizer:
          type: "cvtrainer.data.tensorizer.Tensorizer"
          mapping:
            x:
              type: "cvtrainer.data.tensorizer.ImageTensorizer"
              kwargs: {normalize: imagenet, dtype: float}
            class_id:
              type: "cvtrainer.data.tensorizer.OneHotTensorizer"
              kwargs: {num_classes: 101, dtype: float}
        dataloader_args:
          batch_size: 32
          num_workers: 4
          shuffle: true
      
      val:
        dataset:
          data_path: "data/val"
        transforms:
          backend: "Albumentations"
          transforms:
            - type: "Resize"
              kwargs: {height: 224, width: 224}
        tensorizer:
          type: "cvtrainer.data.tensorizer.Tensorizer"
          mapping:
            x:
              type: "cvtrainer.data.tensorizer.ImageTensorizer"
              kwargs: {normalize: imagenet, dtype: float}
            class_id:
              type: "cvtrainer.data.tensorizer.OneHotTensorizer"
              kwargs: {num_classes: 101, dtype: float}
        dataloader_args:
          batch_size: 32
          shuffle: false
    
    logger:
      - type: "console"
      - type: "file"
        kwargs:
          log_dir: "./logs"
          filename: "train.log"
      - type: "cvtrainer.logger.TensorBoardLogger"
        kwargs:
          log_dir: "./runs"
    
    hooks:
      - type: "cvtrainer.hooks.ProgressBarHook"
      - type: "cvtrainer.hooks.LoggerHook"
        kwargs:
          train_log_freq: 10
          val_log_freq: 1
      - type: "cvtrainer.hooks.LRSchedulerHook"
        kwargs:
          scheduler_type: "CosineAnnealingLR"
          scheduler_kwargs: {T_max: 100}
      - type: "cvtrainer.hooks.EMAHook"
        kwargs: {decay: 0.999}
      - type: "cvtrainer.hooks.CheckpointHook"
        kwargs:
          save_dir: "./checkpoints"
          save_freq: 10
          save_best: true
          metric_name: "val_accuracy"
          mode: "max"
    
    meters:
      - type: "cvtrainer.meters.LossMeter"
      - type: "cvtrainer.meters.AccuracyMeter"
        kwargs: {topk: [1, 5]}
    
    ddp:
      enabled: false
      num_gpus: 1
```

## 配置系统

### !include 标签

引用外部配置文件中的配置：

```yaml
# 引用整个文件
transforms: "!include ../../common/transforms.yaml"

# 引用特定 key
transforms: "!include ../../common/transforms.yaml::aug_medium"
```

### template 字段

使用模板并复写字段（递归 update）：

```yaml
# 完全使用模板
transforms: "!include ../../common/transforms.yaml::aug_medium"

# 复写 backend
transforms:
  template: "!include ../../common/transforms.yaml::aug_medium"
  backend: "Torchvision"

# 完全替换 transforms 列表
transforms:
  template: "!include ../../common/transforms.yaml::aug_medium"
  transforms:
    - type: "Resize"
      kwargs: {height: 224, width: 224}
    - type: "ColorJitter"
      kwargs: {brightness: 0.2, contrast: 0.2}
```

## 架构设计

### 核心组件

CVTrainer 采用极简 Trainer + Hook 扩展的架构设计：

- **Trainer 类**：只负责最基础的业务过程（train/eval/step），不包含任何训练技巧
- **Hook 系统**：16 个调用点，所有训练技巧通过 Hook 实现
- **HookContext**：Hook 间通信上下文
- **日志系统**：支持多种日志后端

### 三段式模型

```
Backbone → Adapter → Head
```

Backbone 提取特征，Adapter 调整通道，Head 输出预测。

## 日志系统

支持四种日志方式：

```yaml
logger:
  - type: "console"                          # 控制台输出
  - type: "file"                             # 文件日志
    kwargs:
      log_dir: "./logs"
      filename: "train.log"
  - type: "cvtrainer.logger.TensorBoardLogger"  # TensorBoard
    kwargs:
      log_dir: "./runs"
  - type: "cvtrainer.logger.WandBLogger"    # Weights & Biases
    kwargs:
      project: "my_project"
```

## Hook 系统

### 内置 Hooks

| Hook | 功能 |
|------|------|
| ProgressBarHook | 实时进度条显示 |
| LoggerHook | 日志记录，支持多种后端 |
| LRSchedulerHook | 学习率调度 |
| EMAHook | 指数移动平均 |
| CheckpointHook | 检查点保存，支持最佳模型 |
| DDPHook | 分布式训练初始化和同步 |

### 自定义 Hook

```python
from cvtrainer.hooks.base import BaseHook

class MyCustomHook(BaseHook):
    def after_step(self, stage):
        loss = stage["context"].get("loss")
        print(f"Step loss: {loss}")
```

## 项目结构

```
CVTrainer/
├── cvtrainer/
│   ├── core/              # 核心配置和构建逻辑
│   ├── models/            # 模型组件
│   │   ├── backbones/     # Backbone 实现
│   │   ├── adapters/      # Adapter 实现
│   │   └── heads/         # Head 实现
│   ├── data/              # 数据引擎
│   │   ├── datasets/      # Dataset 实现
│   │   ├── transforms/    # 数据增强
│   │   ├── tensorizer/    # Tensorizer 实现
│   │   └── collate.py     # Collate 函数
│   ├── optim/             # 优化器和调度器
│   ├── hooks/             # Hook 系统
│   ├── logger/            # 日志系统
│   ├── meters/            # 指标计算
│   ├── loss/              # 损失函数
│   ├── ddp/               # DDP 支持
│   ├── trainer.py         # 极简 Trainer 类
│   └── context.py         # HookContext
├── apps/                  # 用户项目目录
│   └── your_project/
│       ├── model.py       # 模型定义
│       ├── train.yaml     # 训练配置
│       ├── eval.yaml      # 评估配置
│       ├── checkpoints/   # 检查点保存目录
│       └── logs/          # 日志目录
└── configs/
    └── common/            # 通用配置模板
```

## 开发者指南

开发者只能在 `apps/` 目录下进行开发：

```bash
# 创建新项目
mkdir -p apps/my_project
cd apps/my_project

# 创建模型
# 创建 train.yaml

# 运行训练
cvtrainer train --config my_project/train.yaml
```

## License

MIT
