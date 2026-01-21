"""交叉熵损失函数的单元测试"""
import pytest
import torch
from cvtrainer.loss import CrossEntropyLoss


class TestCrossEntropyLoss:
    """测试 CrossEntropyLoss 类"""

    def test_forward_output_shape(self, device):
        """测试前向传播输出形状"""
        loss_fn = CrossEntropyLoss().to(device)
        output = torch.randn(32, 10, device=device)
        target = torch.randint(0, 10, (32,), device=device)

        loss = loss_fn(output, target)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_loss_value(self, device):
        """测试前向传播损失值"""
        loss_fn = CrossEntropyLoss().to(device)

        output = torch.randn(32, 10, device=device)
        target = torch.randint(0, 10, (32,), device=device)

        loss = loss_fn(output, target)

        assert isinstance(loss.item(), float)

    def test_backward(self, device):
        """测试反向传播"""
        loss_fn = CrossEntropyLoss().to(device)

        output = torch.randn(32, 10, device=device, requires_grad=True)
        target = torch.randint(0, 10, (32,), device=device)

        loss = loss_fn(output, target)
        loss.backward()

        assert output.grad is not None
        assert output.grad.shape == output.shape

    def test_perfect_prediction_loss(self, device):
        """测试完美预测时的损失值"""
        loss_fn = CrossEntropyLoss().to(device)

        num_classes = 10
        output = torch.zeros(1, num_classes, device=device)
        output[0, 0] = 100.0
        target = torch.tensor([0], device=device)

        loss = loss_fn(output, target)

        assert loss.item() < 1.0

    def test_different_batch_sizes(self, device):
        """测试不同 batch size"""
        loss_fn = CrossEntropyLoss().to(device)

        for batch_size in [1, 8, 32, 64]:
            output = torch.randn(batch_size, 10, device=device)
            target = torch.randint(0, 10, (batch_size,), device=device)

            loss = loss_fn(output, target)
            assert loss.dim() == 0
