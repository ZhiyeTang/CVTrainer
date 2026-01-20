# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2025-01-18

### 新增

#### 核心框架
- 配置驱动的训练框架
- YAML 配置解析系统
- `!include` 标签支持外部文件引用
- `template` 字段支持模板复写（递归深度合并）

#### 模型系统
- 三段式模型架构（Backbone → Adapter → Head）
- BaseModel、BaseBackbone、BaseAdapter、BaseHead 基类
- 内置 Backbone：ResNet18、ResNet50
- 内置 Adapter：Identity、Conv、Linear
- 内置 Head：MulticlassClassifier、MultiLabelClassifier

#### 数据引擎
- BaseDataAdapter 基类
- ImageFolderDataAdapter 实现
- Transforms 系统：
  - BaseTransform 基类
  - Albumentations Backend
  - Torchvision Backend
  - transforms 配置解析
- Tensorizer 系统：
  - BaseTensorizer 基类
  - Tensorizer Composite
  - ImageTensorizer
  - OneHotTensorizer
  - LongTensorizer
  - FloatTensorizer
  - MaskTensorizer
  - BBoxTensorizer
  - LabelsTensorizer
- Collate 函数：
  - smart_collate_fn（自动 padding）
  - default_collate_fn（固定长度）

#### 训练引擎
- Hook 系统：
  - BaseHook 基类（16 个调用点）
  - LRSchedulerHook
  - EMAHook
  - CheckpointHook
- Meter 系统：
  - BaseMeter 基类
  - AccuracyMeter
  - LossMeter
- Optimizer 工厂（支持分层学习率）
- Scheduler 工厂

#### 分布式训练
- DDP 工具函数
- torchrun launcher

#### 其他
- CLI 入口（argparse）
- Builder 函数
- Stage dataclass
- 示例应用：food_classifier
- 示例配置：transforms.yaml, tensorizers.yaml, train.yaml
- 测试：config, models
- 文档：架构、API、最佳实践、贡献指南

## [未发布]

### 计划中

#### 0.2.0
- [ ] Trainer 完整实现
- [ ] MobileNet Backbone
- [ ] ViT Backbone
- [ ] 更多 Adapter
- [ ] 更多 Head（检测、分割）
- [ ] TensorBoard Hook
- [ ] 混合精度训练 Hook
- [ ] 梯度累积 Hook
- [ ] 早停 Hook
- [ ] 学习率预热 Hook

#### 0.3.0
- [ ] 检测任务支持
- [ ] 分割任务支持
- [ ] 关键点检测支持
- [ ] 自动混合精度集成
- [ ] 模型压缩（剪枝、量化）
- [ ] 更多 Optimizer（SGD, AdamW）
- [ ] 更多 Scheduler（ReduceLROnPlateau, CosineAnnealingWarmRestarts）

#### 0.4.0
- [ ] DeepSpeed 集成
- [ ] FSDP 集成
- [ ] torch.compile 集成
- [ ] 模型导出（ONNX, TorchScript）
- [ ] 模型量化
- [ ] 模型部署工具

## 版本说明

### 语义化版本规则

- **主版本号（MAJOR）**：不兼容的 API 变更
- **次版本号（MINOR）**：向后兼容的功能新增
- **修订号（PATCH）**：向后兼容的问题修正

### 发布流程

1. 更新版本号（`pyproject.toml`）
2. 更新更新日志（`CHANGELOG.md`）
3. 创建 Git tag
4. 发布到 PyPI

## 贡献者

感谢所有贡献者！

## 许可证

MIT
