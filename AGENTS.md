# CVTrainer Agent Guide

## Build/Lint/Test Commands

```bash
# Install dependencies
pip install -e .
pip install -r requirements.txt

# Format code
black cvtrainer/ tests/ apps/

# Run all tests
pytest

# Run single test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::test_include_yaml

# Run with verbose output
pytest -v
```

## Code Style Guidelines

### Imports
- Standard library imports first, third-party imports next, local imports last
- Use `from typing import Dict, Any, Optional, List, Tuple` for type hints
- Avoid wildcard imports
- Example:
```python
import os
from typing import Dict, Any
import torch
from .base import BaseAdapter
```

### Formatting
- Use Black formatter with line-length 100
- Target Python version: 3.10+
- Run `black cvtrainer/ tests/ apps/` before committing

### Type Hints
- Add type hints to all function parameters and return values
- Use forward references in quotes for circular imports: `"BaseBackbone"`
- Common types: `Dict[str, Any]`, `torch.Tensor`, `Optional[...]`

### Naming Conventions
- Classes: PascalCase (BaseModel, BaseHook, ResNet18Backbone)
- Functions/Methods: snake_case (build_stage, import_module_str)
- Private functions: prefix with underscore (_build_model)
- Variables: snake_case (backend_name, result)
- Constants: UPPER_CASE (if used)

### Error Handling
- Use `ValueError` for invalid configuration or unknown options
- Format: `raise ValueError(f"Unknown backend: {backend_name}")`
- Keep error messages descriptive and Chinese

### Docstrings
- Use Chinese docstrings throughout the codebase
- Format:
```python
def function_name(arg: str) -> Dict[str, Any]:
    """
    函数描述

    Args:
        arg: 参数说明

    Returns:
        返回值说明
    """
```

### Comments
- Add docstrings instead of inline comments
- Keep comments minimal and in Chinese

### Architecture Patterns
- Use configuration-driven architecture with YAML files
- Base classes inherit from ABC and use `@abstractmethod` decorators
- Use `@property` for computed attributes (e.g., `backbone_channel`)
- Three-tier model architecture: Backbone → Adapter → Head
- Hook system with 16 callback points defined in BaseHook
- Use `importlib.import_module` for dynamic module loading

### Project Structure
```
cvtrainer/
├── core/          # Configuration and building logic
├── models/        # Model components (backbones, adapters, heads)
├── data/          # Data engine (datasets, transforms, tensorizer)
├── optim/         # Optimizers and schedulers
├── hooks/         # Hook system
├── meters/        # Metrics calculation
├── loss/          # Loss functions
└── ddp/           # DDP support
```

### Testing
- Test files: `tests/test_*.py`
- Test functions: `test_*`
- Use pytest assertions
- Mock external dependencies when needed

### Configuration
- Use YAML configuration files
- Support `!include` for external file references
- Support `template` field for overriding templates
- All fields must be explicitly configured
