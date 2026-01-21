"""pytest 全局配置和 fixtures"""
import pytest
import torch
from typing import Generator


@pytest.fixture(scope="session")
def device() -> torch.device:
    """返回测试设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def random_seed() -> Generator[None, None, None]:
    """设置随机种子，确保测试可复现"""
    torch.manual_seed(42)
    yield
    torch.manual_seed(torch.initial_seed())


@pytest.fixture
def requires_cuda():
    """跳过非CUDA测试"""
    if not torch.cuda.is_available():
        pytest.skip("需要CUDA支持")


@pytest.fixture
def requires_multiple_gpus():
    """跳过单GPU测试"""
    if torch.cuda.device_count() < 2:
        pytest.skip("需要至少2个GPU")


@pytest.fixture
def tensor_device(device):
    """返回带设备的tensor fixture"""
    def _create_tensor(*size, requires_grad=False):
        return torch.randn(*size, device=device, requires_grad=requires_grad)
    return _create_tensor


# 导入 fixtures
from tests.fixtures.data_fixtures import *
from tests.fixtures.model_fixtures import *
from tests.fixtures.trainer_fixtures import *
