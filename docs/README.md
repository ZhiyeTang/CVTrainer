# CVTrainer 文档

欢迎来到 CVTrainer 详细文档！

## 快速导航

- **[README.md](../README.md)** - 项目简介和快速开始
- **[架构设计](./architecture.md)** - 框架架构和设计原理
- **[API 参考](./api.md)** - 完整 API 文档
- **[最佳实践](./best_practices.md)** - 使用技巧和优化建议
- **[贡献指南](./contributing.md)** - 如何参与项目开发
- **[更新日志](./changelog.md)** - 版本变更历史

## 文档分类

### 概念理解

- [架构设计](./architecture.md)
  - 设计原则
  - 核心架构（Trainer + Hook 扩展）
  - 模块详解
  - 设计决策

### 使用指南

- [API 参考](./api.md)
  - 配置 API
  - Trainer API
  - 模型 API
  - 数据引擎 API
  - Hook API
  - Logger API
  - Meter API
  - 优化器 API
  - DDP API

- [最佳实践](./best_practices.md)
  - 配置管理
  - 模型设计
  - 数据处理
  - 训练技巧
  - 日志配置
  - 调试技巧
  - 常见问题

### 开发相关

- [贡献指南](./contributing.md)
  - 如何贡献
  - 开发指南
  - 代码规范

## 快速开始

### 1. 安装

```bash
pip install -e .
```

### 2. 基本使用

```bash
# 训练
cvtrainer train --config apps/my_project/train.yaml

# 从 checkpoint 恢复训练
cvtrainer train --config apps/my_project/train.yaml --resume checkpoints/checkpoint_epoch_10.pth

# 评估
cvtrainer eval --config apps/my_project/eval.yaml --checkpoint checkpoints/best_model.pth

# DDP 训练
cvtrainer train --config apps/my_project/train.yaml --num_gpus 4
```

### 3. 配置示例

```yaml
stages:
  - name: "train"
    epochs: 100
    model: "apps.my_project.model.MyModel"
    model_args: {num_classes: 101}
    logger:
      - type: "console"
      - type: "file"
        kwargs:
          log_dir: "./logs"
    hooks:
      - type: "cvtrainer.hooks.ProgressBarHook"
      - type: "cvtrainer.hooks.LoggerHook"
        kwargs: {train_log_freq: 10}
      - type: "cvtrainer.hooks.CheckpointHook"
        kwargs:
          save_dir: "./checkpoints"
          save_freq: 10
          save_best: true
          metric_name: "val_accuracy"
```

## 架构设计

### Trainer + Hook 扩展模式

CVTrainer 采用极简 Trainer + Hook 扩展的架构设计：

```
┌─────────────────────────────────────────────────────────┐
│                        Trainer                           │
│  ┌─────────────────────────────────────────────────────┐│
│  │ train() / eval() / train_step() / eval_step()       ││
│  └─────────────────────────────────────────────────────┘│
│                          ↑                              │
│                    call_hooks()                         │
└─────────────────────────────────────────────────────────┘
                           ↑
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ProgressBarHook    CheckpointHook      EMAHook
        │                  │                  │
        ↓                  ↓                  ↓
   Console/File      检查点保存         EMA 模型更新
   TensorBoard       最佳模型保存       评估时切换
   WandB             配置哈希           (before_eval)
```

### 核心设计原则

1. **Trainer 极简化**：只包含最基础的业务过程
2. **所有扩展通过 Hook**：DDP、EMA、Checkpoint、Logger 都是 Hook
3. **HookContext 通信**：Hook 间通过 context 共享状态
4. **配置驱动**：所有组件通过配置文件组装

## 学习路径

### 初学者

1. 阅读 [README.md](../README.md)
2. 跟随快速开始示例
3. 阅读 [架构设计](./architecture.md) 了解框架原理
4. 参考 [最佳实践](./best_practices.md) 调优模型

### 进阶用户

1. 深入 [API 参考](./api.md) 了解所有 API
2. 学习 [最佳实践](./best_practices.md) 优化技巧
3. 自定义组件（模型、Hook、Tensorizer）
4. 配置日志系统（Console、File、TensorBoard、WandB）

### 开发者

1. 阅读 [架构设计](./architecture.md) 理解设计原理
2. 阅读 [贡献指南](./contributing.md) 了解开发流程
3. 参与开源贡献

## 常见问题

### 如何添加新模型？

参见 [最佳实践 - 模型设计](./best_practices.md#模型设计)

### 如何自定义 Hook？

参见 [最佳实践 - 自定义 Hook](./best_practices.md#自定义-hook)

### 如何配置日志？

参见 [最佳实践 - 日志配置](./best_practices.md#日志配置)

### 如何使用 DDP？

参见 [README - DDP 训练](../README.md#ddp-训练)

### 如何解决常见问题？

参见 [最佳实践 - 常见问题解决](./best_practices.md#常见问题解决)

## 反馈与贡献

- 提交问题：[GitHub Issues](../../issues)
- 贡献代码：[贡献指南](./contributing.md)

## 许可证

MIT License
