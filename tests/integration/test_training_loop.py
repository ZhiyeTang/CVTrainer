"""训练流程的集成测试 - 使用 CVTrainer 数据系统"""
import pytest
import torch
from cvtrainer.trainer import Trainer
from cvtrainer.meters import LossMeter
from cvtrainer.loss import CrossEntropyLoss
from tests.fixtures.data_fixtures import MockTensorDataset


class TestTrainingLoopWithDataAdapter:
    """使用 CVTrainer DataAdapter 测试完整训练流程"""

    def test_single_epoch_training(self, device, small_model, mock_tensor_dataloader):
        """测试单个epoch的训练"""
        dataloader = mock_tensor_dataloader(batch_size=10, num_samples=20, num_classes=10)

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

        trainer = Trainer(
            model=small_model,
            criterion=criterion,
            optimizer=optimizer,
            dataloaders={"train": dataloader},
            hooks=[],
            meters={"loss": LossMeter()},
            device=device
        )

        trainer.epoch = 0
        trainer._train_epoch()

        assert trainer.epoch == 0
        assert trainer.step > 0

    def test_eval_mode(self, device, small_model, mock_tensor_dataloader):
        """测试评估模式"""
        dataloader = mock_tensor_dataloader(batch_size=10, num_samples=20, num_classes=10)

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

        trainer = Trainer(
            model=small_model,
            criterion=criterion,
            optimizer=optimizer,
            dataloaders={"val": dataloader},
            hooks=[],
            meters={"loss": LossMeter()},
            device=device
        )

        trainer.eval("val")

        assert trainer.model.training == False

    def test_train_eval_mode_switch(self, device, small_model, mock_tensor_dataloader):
        """测试训练/评估模式切换"""
        train_loader = mock_tensor_dataloader(batch_size=10, num_samples=20, num_classes=10)
        val_loader = mock_tensor_dataloader(batch_size=10, num_samples=20, num_classes=10)

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

        trainer = Trainer(
            model=small_model,
            criterion=criterion,
            optimizer=optimizer,
            dataloaders={"train": train_loader, "val": val_loader},
            hooks=[],
            meters={"loss": LossMeter()},
            device=device
        )

        assert trainer.model.training == True

        trainer.eval("val")

        assert trainer.model.training == False

        trainer._train_epoch()

        assert trainer.model.training == True

    def test_meter_tracking_during_training(self, device, small_model, mock_tensor_dataloader):
        """测试训练过程中meter的跟踪"""
        dataloader = mock_tensor_dataloader(batch_size=10, num_samples=20, num_classes=10)

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

        loss_meter = LossMeter()

        trainer = Trainer(
            model=small_model,
            criterion=criterion,
            optimizer=optimizer,
            dataloaders={"train": dataloader},
            hooks=[],
            meters={"loss": loss_meter},
            device=device
        )

        trainer._train_epoch()

        loss_value = loss_meter.get_value()
        assert isinstance(loss_value, float)
        assert loss_value >= 0

    def test_trainer_with_custom_meters(self, device, small_model, mock_tensor_dataloader):
        """测试带自定义meter的训练"""
        from cvtrainer.meters import F1Meter

        dataloader = mock_tensor_dataloader(batch_size=10, num_samples=20, num_classes=10)

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)

        trainer = Trainer(
            model=small_model,
            criterion=criterion,
            optimizer=optimizer,
            dataloaders={"train": dataloader},
            hooks=[],
            meters={"loss": LossMeter(), "f1": F1Meter(num_classes=10)},
            device=device
        )

        trainer._train_epoch()

        meters = trainer.context.get("meters")
        assert "loss" in meters
        assert "f1" in meters


class TestCheckpointResume:
    """测试检查点保存和恢复"""

    def test_state_dict_and_load(self, sample_trainer):
        """测试状态保存和加载"""
        sample_trainer.epoch = 5
        sample_trainer.step = 100
        sample_trainer.context.set("loss", 0.5)

        state = sample_trainer.state_dict()

        assert state["epoch"] == 5
        assert state["step"] == 100

        new_trainer = sample_trainer
        new_trainer.load_state_dict(state)

        assert new_trainer.epoch == 5
        assert new_trainer.step == 100

    def test_model_state_dict(self, sample_trainer):
        """测试模型状态保存"""
        original_params = {name: param.clone() for name, param in sample_trainer.model.named_parameters()}

        state = sample_trainer.state_dict()
        loaded_state = sample_trainer.state_dict()

        for name, param in sample_trainer.model.named_parameters():
            assert torch.equal(loaded_state["model"][name], original_params[name])
