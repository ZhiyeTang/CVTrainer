# CIFAR-10 Demo Project

这是一个基于CVTrainer框架的CIFAR-10图像分类demo项目。

## 项目结构

```
apps/cifar10_demo/
├── __init__.py          # 模块初始化
├── model.py             # CIFAR-10模型定义
└── dataset.py           # CIFAR-10数据集适配器

configs/apps/cifar10_demo/
└── train.yaml           # 训练配置文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. 运行训练

```bash
cvtrainer --config configs/apps/cifar10_demo/train.yaml
```

## 模型架构

CIFAR-10模型采用三段式架构：

- **Backbone**: ResNet18 (无预训练权重)
- **Adapter**: LinearAdapter (降维到256维)
- **Head**: MulticlassClassifier (10类分类)

## 数据集

CIFAR-10数据集会自动下载到`./data/cifar10`目录下。

- 训练集：50,000张图片
- 测试集：10,000张图片
- 图像大小：32x32
- 类别数：10

## 数据增强

训练时使用以下数据增强：
- Resize (32x32)
- RandomHorizontalFlip (p=0.5)
- RandomRotate90 (p=0.5)

验证时只使用Resize。

## 训练配置

- **Epochs**: 100
- **Batch Size**: 128
- **Optimizer**: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- **LR Scheduler**: MultiStepLR (milestones=[50, 75], gamma=0.1)
- **Checkpoint**: 每10个epoch保存一次到`./checkpoints/cifar10`

## 自定义

如需修改模型或训练配置，可以编辑以下文件：

- `apps/cifar10_demo/model.py` - 修改模型架构
- `configs/apps/cifar10_demo/train.yaml` - 修改训练配置

## 示例输出

训练过程会显示以下指标：
- Loss
- Accuracy (Top-1, Top-5)
- 学习率变化

检查点文件会保存到`./checkpoints/cifar10/`目录。
