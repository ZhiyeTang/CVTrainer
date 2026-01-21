"""Hook基类的单元测试"""
import pytest
from cvtrainer.hooks.base import BaseHook
from cvtrainer.context import HookContext


class TestBaseHook:
    """测试 BaseHook 类"""

    def test_base_hook_methods_exist(self):
        """测试基类方法存在"""
        hook = BaseHook()

        assert hasattr(hook, "before_train")
        assert hasattr(hook, "after_train")
        assert hasattr(hook, "before_epoch")
        assert hasattr(hook, "after_epoch")
        assert hasattr(hook, "before_step")
        assert hasattr(hook, "after_step")
        assert hasattr(hook, "before_forward")
        assert hasattr(hook, "after_forward")
        assert hasattr(hook, "before_backward")
        assert hasattr(hook, "after_backward")
        assert hasattr(hook, "before_optimize")
        assert hasattr(hook, "after_optimize")
        assert hasattr(hook, "before_eval")
        assert hasattr(hook, "after_eval")
        assert hasattr(hook, "before_save")
        assert hasattr(hook, "after_save")

    def test_base_hook_methods_return_none(self):
        """测试基类方法返回None"""
        hook = BaseHook()
        context = HookContext()

        stage = {"context": context, "trainer": None, "epoch": 0, "phase": "train"}

        assert hook.before_train(stage) is None
        assert hook.after_train(stage) is None
        assert hook.before_epoch(stage) is None
        assert hook.after_epoch(stage) is None
        assert hook.before_step(stage) is None
        assert hook.after_step(stage) is None


class TestHookContext:
    """测试 HookContext 类"""

    def test_context_set_and_get(self):
        """测试上下文设置和获取"""
        context = HookContext()

        context.set("key1", "value1")
        context.set("key2", 42)

        assert context.get("key1") == "value1"
        assert context.get("key2") == 42

    def test_context_get_nonexistent(self):
        """测试获取不存在的key"""
        context = HookContext()

        assert context.get("nonexistent") is None
        assert context.get("nonexistent", "default") == "default"

    def test_context_set_none(self):
        """测试设置None值"""
        context = HookContext()

        context.set("key", None)
        assert context.get("key") is None

    def test_context_overwrite(self):
        """测试覆盖值"""
        context = HookContext()

        context.set("key", "value1")
        context.set("key", "value2")

        assert context.get("key") == "value2"
