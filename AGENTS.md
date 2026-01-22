# CVTrainer Agent Guide

## Build/Lint/Test Commands

```bash
# Install dependencies
mamba run -n dev pip install -e .
mamba run -n dev pip install -r requirements.txt

# Format code (Black, line-length 100)
mamba run -n dev black cvtrainer/ tests/ apps/

# Run all tests
mamba run -n dev python -m pytest

# Run single test file
mamba run -n dev python -m pytest tests/unit/core/test_trainer.py -v

# Run specific test
mamba run -n dev python -m pytest tests/unit/core/test_trainer.py::test_flatten_single_value -v

# Run with verbose and short traceback
mamba run -n dev python -m pytest -v --tb=short
```

## Code Style Guidelines

### Imports
- Standard library imports first, third-party imports next, local imports last
- Use `from typing import Dict, Any, Optional, List, Tuple, Union` for type hints
- Import `Number` from `numbers` for numeric types
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

tests/
├── conftest.py    # Global pytest config + fixtures
├── fixtures/      # Shared test fixtures
├── unit/          # Unit tests
└── integration/   # Integration tests
```

### Testing
- Test files: `tests/unit/test_*.py` and `tests/integration/test_*.py`
- Test functions: `test_*`
- Test fixtures: `tests/fixtures/*.py`
- Use pytest assertions
- Mock external dependencies when needed
- Test data: Use `MockTensorDataset` inheriting from `BaseDataAdapter` for integration tests

### Meter Implementation Patterns
- Meters must return `Union[Number, Dict[str, Number]]` from `get_value()`
- Single-value meters (LossMeter, SegmentationIoUMeter): return `Number`
- Multi-value meters (AccuracyMeter, DetectionMapMeter): return `Dict[str, Number]`
- Keys in dict should clearly indicate metric meaning (e.g., `accuracy_1`, `map_50`)

### Configuration
- Use YAML configuration files
- Support `!include` for external file references
- Support `template` field for overriding templates
- All fields must be explicitly configured

### Important Patterns
- ResNet backbones: Return 4D feature maps [B, C, H, W] (not 2D [B, C])
- Classifier heads: Support 4D inputs by adding global average pooling
- Trainer._flatten_meters(): Intelligently flattens dict values with prefix matching
  - Single values: `{"loss": 0.5}`
  - Dicts with matching prefix: `{"map": 0.5, "map_50": 0.6}`
  - Dicts without prefix: `{"classification/precision": 0.8}`
- DataAdapter: Implement `_load_sample()` returning dict with `"x"` and `"target"` keys

## Testing Best Practices
- Fixtures should be in `tests/fixtures/` and imported in `conftest.py`
- Use `device` fixture for testing on CPU/CUDA
- Use `random_seed` fixture for reproducibility
- Integration tests should use `MockTensorDataset` inheriting from `BaseDataAdapter`
- Write tests for both positive and negative cases
