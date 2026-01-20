import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader(yaml.SafeLoader):
    """支持 !include 标签的 YAML Loader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loading_files: set = set()


def include_constructor(loader: ConfigLoader, node: yaml.Node) -> Any:
    """
    处理 !include 标签

    格式: !include <file_path>::<key>
    示例: !include ./common/transforms.yaml::aug_medium
    """
    value = loader.construct_scalar(node)

    if "::" in value:
        file_path, key = value.split("::", 1)
    else:
        file_path, key = value, None

    if loader.stream and loader.stream.name:
        current_dir = Path(loader.stream.name).parent
    else:
        current_dir = Path.cwd()

    full_path = (current_dir / file_path).resolve()

    with open(full_path, "r") as f:
        data = load_yaml(f.read(), full_path)

    return data[key] if key else data


ConfigLoader.add_constructor("!include", include_constructor)


def deep_update(target: Dict, source: Dict) -> Dict:
    """
    递归更新字典（深度合并）

    Args:
        target: 目标字典
        source: 源字典

    Returns:
        更新后的字典
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            target[key] = deep_update(target[key], value)
        else:
            target[key] = value
    return target


def resolve_template_field(config: Any) -> Any:
    """
    处理配置中的 template 字段，实现模板复写

    规则：
    1. 加载 template 指定的模板
    2. 用当前配置深度合并模板中的字段
    3. 删除 template 字段
    4. 递归处理嵌套的 template 字段
    """
    if not isinstance(config, dict):
        return config

    result = config.copy()

    if "template" in result:
        template_config = result.pop("template")

        if isinstance(template_config, dict):
            template_config = resolve_template_field(template_config)
            result = deep_update(template_config, result)
        else:
            result = template_config

    for key, value in result.items():
        result[key] = resolve_template_field(value)

    return result


def load_yaml(content: str, file_path: Path = None) -> Dict:
    """
    加载 YAML 配置

    Args:
        content: YAML 内容
        file_path: 文件路径（用于解析相对路径）

    Returns:
        解析后的配置字典
    """
    if file_path:
        config = yaml.load(content, Loader=ConfigLoader)
    else:
        config = yaml.safe_load(content)

    config = resolve_template_field(config)
    return config


def load_yaml_file(file_path: str) -> Dict:
    """
    从文件加载 YAML 配置

    Args:
        file_path: 配置文件路径

    Returns:
        解析后的配置字典
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return load_yaml(f.read(), path)
