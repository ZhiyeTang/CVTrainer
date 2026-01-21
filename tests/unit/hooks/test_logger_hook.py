"""Logger Hook的单元测试"""

import pytest
from unittest.mock import MagicMock, patch
from cvtrainer.hooks.logger import LoggerHook
from cvtrainer.context import HookContext


class TestLoggerHook:
    """测试 LoggerHook 类"""

    def test_logger_hook_initialization(self):
        """测试初始化"""
        hook = LoggerHook(train_log_freq=10, val_log_freq=1)

        assert hook.train_log_freq == 10
        assert hook.val_log_freq == 1
        assert hook.loggers == []
        assert hook.logged_values == {}

    def test_logger_hook_add_logger(self):
        """测试添加logger"""
        hook = LoggerHook()
        mock_logger = MagicMock()

        hook.add_logger(mock_logger)

        assert len(hook.loggers) == 1
        assert hook.loggers[0] == mock_logger

    def test_logger_hook_add_multiple_loggers(self):
        """测试添加多个logger"""
        hook = LoggerHook()
        mock_logger1 = MagicMock()
        mock_logger2 = MagicMock()

        hook.add_logger(mock_logger1)
        hook.add_logger(mock_logger2)

        assert len(hook.loggers) == 2

    def test_logger_hook_after_step(self):
        """测试 after_step 方法"""
        hook = LoggerHook(train_log_freq=10, val_log_freq=1)
        mock_logger = MagicMock()
        hook.add_logger(mock_logger)

        context = HookContext()
        context.set("loss", 0.5)
        context.set("meters", {"accuracy": 95.0})

        stage = {"context": context, "epoch": 0, "step": 10, "phase": "train"}

        hook.after_step(stage)

        mock_logger.log_metrics.assert_called_once()
        call_args = mock_logger.log_metrics.call_args[0][0]
        assert call_args["epoch"] == 0
        assert call_args["step"] == 10
        assert call_args["phase"] == "train"
        assert call_args["loss"] == 0.5

    def test_logger_hook_after_step_freq_skip(self):
        """测试 after_step 频率跳过"""
        hook = LoggerHook(train_log_freq=10, val_log_freq=1)
        mock_logger = MagicMock()
        hook.add_logger(mock_logger)

        context = HookContext()
        context.set("loss", 0.5)

        stage = {"context": context, "epoch": 0, "step": 5, "phase": "train"}

        hook.after_step(stage)
        mock_logger.log_metrics.assert_not_called()

    def test_logger_hook_after_epoch(self):
        """测试 after_epoch 方法"""
        hook = LoggerHook()
        mock_logger = MagicMock()
        hook.add_logger(mock_logger)

        context = HookContext()
        context.set("meters", {"accuracy": 95.0, "loss": 0.5})

        stage = {"context": context, "epoch": 0, "phase": "val"}

        hook.after_epoch(stage)

        mock_logger.log_epoch_summary.assert_called_once()

    def test_logger_hook_state_dict(self):
        """测试 state_dict 方法"""
        hook = LoggerHook()
        hook.logged_values["last_step"] = 100

        state = hook.state_dict()

        assert "logged_values" in state
        assert state["logged_values"]["last_step"] == 100

    def test_logger_hook_load_state_dict(self):
        """测试 load_state_dict 方法"""
        hook = LoggerHook()
        state = {"logged_values": {"last_step": 50}}

        hook.load_state_dict(state)

        assert hook.logged_values["last_step"] == 50
