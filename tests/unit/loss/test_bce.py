"""BCE损失函数的单元测试"""
import pytest
import torch
from cvtrainer.loss import BCELoss


class TestBCELoss:
    """测试 BCELoss 类"""

    def test_forward_output_shape(self, device):
        """测试前向传播输出形状"""
        loss_fn = BCELoss().to(device)
        output = torch.randn(32, 10, device=device)
        target = torch.randint(0, 2, (32, 10), device=device).float()

        loss = loss_fn(output, target)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_loss_value(self, device):
        """测试前向传播损失值"""
        loss_fn = BCELoss().to(device)

        output = torch.randn(32, 10, device=device)
        target = torch.randint(0, 2, (32, 10), device=device).float()

        loss = loss_fn(output, target)

        assert isinstance(loss.item(), float)

    def test_backward(self, device):
        """测试反向传播"""
        loss_fn = BCELoss().to(device)

        output = torch.randn(32, 10, device=device, requires_grad=True)
        target = torch.randint(0, 2, (32, 10), device=device).float()

        loss = loss_fn(output, target)
        loss.backward()

        assert output.grad is not None
        assert output.grad.shape == output.shape

    def test_perfect_prediction_loss(self, device):
        """测试完美预测时的损失值"""
        loss_fn = BCELoss().to(device)

        output = torch.randn(32, 10, device=device)
        target = (torch.sigmoid(output) > 0.5).float()

        loss = loss_fn(output, target)

        assert loss.item() < 1.0

    def test_different_batch_sizes(self, device):
        """测试不同 batch size"""
        loss_fn = BCELoss().to(device)

        for batch_size in [1, 8, 32, 64]:
            output = torch.randn(batch_size, 10, device=device)
            target = torch.randint(0, 2, (batch_size, 10), device=device).float()

            loss = loss_fn(output, target)
            assert loss.dim() == 0

    def test_multilabel_scenario(self, device):
        """测试多标签场景"""
        loss_fn = BCELoss().to(device)

        num_samples = 32
        num_classes = 10
        output = torch.randn(num_samples, num_classes, device=device)
        target = torch.zeros(num_samples, num_classes, device=device)
        target[:, :3] = 1.0

        loss = loss_fn(output, target)

        assert loss.dim() == 0
        assert isinstance(loss.item(), float)
