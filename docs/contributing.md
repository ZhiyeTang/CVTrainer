# 贡献指南

感谢你对 CVTrainer 的关注！我们欢迎任何形式的贡献。

## 如何贡献

### 1. 报告问题

如果你发现了 bug 或有新功能建议，请：

1. 搜索现有的 [Issues](../../issues)
2. 如果不存在，创建新的 Issue
3. 提供详细的信息：
   - 问题描述
   - 复现步骤
   - 预期行为
   - 实际行为
   - 环境信息（Python 版本、依赖版本等）

### 2. 提交代码

#### Fork 项目

1. Fork CVTrainer 仓库
2. Clone 你的 fork：
   ```bash
   git clone https://github.com/your-username/CVTrainer.git
   cd CVTrainer
   ```

#### 创建分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

#### 开发

1. 安装开发依赖：
   ```bash
   pip install -e ".[dev]"
   ```

2. 运行测试：
   ```bash
   pytest tests/
   ```

3. 代码格式化：
   ```bash
   black .
   ```

#### 提交

```bash
git add .
git commit -m "feat: add new feature"  # 或 "fix: fix bug"
```

**提交信息格式**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建或工具相关

#### 推送和创建 PR

```bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request。

## 开发指南

### 项目结构

```
cvtrainer/
├── core/              # 核心配置和构建逻辑
│   ├── config.py       # YAML 解析（!include, template）
│   ├── stage.py        # Stage dataclass
│   ├── builder.py      # build_stage 函数
│   └── trainer.py      # Trainer 类（待实现）
├── models/            # 模型组件
│   ├── backbones/      # Backbone 实现
│   ├── adapters/       # Adapter 实现
│   └── heads/          # Head 实现
├── data/              # 数据引擎
│   ├── datasets/       # Dataset 实现
│   ├── transforms/     # 数据增强
│   ├── tensorizer/     # Tensorizer 实现
│   └── collate.py      # Collate 函数
├── optim/             # 优化器和调度器
├── hooks/             # Hook 系统
├── meters/            # 指标计算
├── loss/              # 损失函数
└── ddp/               # DDP 支持
```

### 添加新 Backbone

1. 在 `cvtrainer/models/backbones/` 创建新文件：
   ```python
   # my_backbone.py
   import torch
   import torch.nn as nn
   from ..base import BaseBackbone

   class MyBackbone(BaseBackbone):
       def __init__(self, pretrained=True):
           super().__init__()
           self.model = my_model(pretrained)
           self._backbone_channel = 512
           self.model.fc = nn.Identity()
       
       @property
       def backbone_channel(self):
           return self._backbone_channel
       
       def forward(self, x):
           return self.model(x)
   ```

2. 在 `cvtrainer/models/backbones/__init__.py` 导出：
   ```python
   from .my_backbone import MyBackbone

   __all__ = [..., "MyBackbone"]
   ```

3. 添加测试：
   ```python
   # tests/test_backbones.py
   from cvtrainer.models.backbones import MyBackbone

   def test_my_backbone():
       backbone = MyBackbone(pretrained=False)
       x = torch.randn(2, 3, 224, 224)
       output = backbone(x)
       assert output.shape[1] == backbone.backbone_channel
   ```

### 添加新 Adapter

1. 在 `cvtrainer/models/adapters/` 创建新文件
2. 实现 `BaseAdapter` 接口
3. 在 `__init__.py` 导出
4. 添加测试

### 添加新 Tensorizer

1. 在 `cvtrainer/data/tensorizer/` 创建新文件
2. 实现 `BaseTensorizer` 接口：
   ```python
   # my_tensorizer.py
   from .base import BaseTensorizer
   import torch

   class MyTensorizer(BaseTensorizer):
       def __call__(self, value):
           return torch.tensor(value, dtype=torch.float32)
   ```

3. 在 `__init__.py` 导出
4. 添加测试

### 添加新 Hook

1. 在 `cvtrainer/hooks/` 创建新文件
2. 实现 `BaseHook` 接口：
   ```python
   # my_hook.py
   from .base import BaseHook

   class MyHook(BaseHook):
       def __init__(self, param1, param2):
           self.param1 = param1
           self.param2 = param2
       
       def after_step(self, stage):
           # 自定义逻辑
           pass
   ```

3. 在 `__init__.py` 导出
4. 添加测试

## 代码规范

### Python 版本

- 支持 Python 3.10+
- 使用 Python 3.10 特性（match-case 等）

### 代码格式

- 使用 Black 格式化
- 最大行长度：100 字符
- 使用类型注解（Type Hints）

### 文档字符串

使用 Google 风格：

```python
def build_optimizer(model, config):
    """
    构建优化器，支持分层学习率
    
    Args:
        model (nn.Module): 模型
        config (Dict): 优化器配置
    
    Returns:
        Optimizer: 优化器实例
    
    Raises:
        ValueError: 配置错误时抛出
    """
    pass
```

### 注释

- 不添加无意义的注释
- 复杂逻辑添加注释说明
- 使用中文注释

## 测试

### 运行所有测试

```bash
pytest tests/
```

### 运行特定测试

```bash
pytest tests/test_config.py
pytest tests/test_models.py -v
```

### 测试覆盖率

```bash
pytest --cov=cvtrainer tests/
```

## 文档

### 更新文档

如果添加了新功能，请更新相应文档：

1. **API 参考**：添加新的类和函数
2. **最佳实践**：添加使用示例
3. **README.md**：添加快速开始示例
4. **更新日志**：记录变更

### 文档风格

- 使用中文
- 提供代码示例
- 简洁明了

## Pull Request 检查清单

提交 PR 前，请确保：

- [ ] 代码通过所有测试
- [ ] 代码已使用 Black 格式化
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 提交信息遵循规范
- [ ] PR 描述清晰说明了变更内容

## 发布流程

1. 更新版本号（`pyproject.toml`）
2. 更新更新日志（`CHANGELOG.md`）
3. 创建 Git tag
4. 发布到 PyPI

## 许可证

提交代码即表示你同意你的代码使用 MIT 许可证发布。

## 联系方式

- GitHub Issues: [提交问题](../../issues)
- 邮箱: [项目邮箱](mailto:project@example.com)

## 致谢

感谢所有贡献者的贡献！

## 行为准则

- 保持友善和专业
- 尊重不同观点
- 专注于改进项目
- 避免人身攻击

## 安全问题

如果你发现安全问题，请不要公开 Issue，请直接联系维护者。

再次感谢你的贡献！
