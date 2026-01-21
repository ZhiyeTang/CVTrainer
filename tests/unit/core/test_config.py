import pytest
from cvtrainer.core import load_yaml


def test_include_yaml():
    yaml_content = """
    transforms:
      backend: "Albumentations"
      transforms:
        - type: "Resize"
          kwargs: {height: 256, width: 256}
    """

    result = load_yaml(yaml_content)
    assert result["transforms"]["backend"] == "Albumentations"
    assert result["transforms"]["transforms"][0]["type"] == "Resize"


def test_template_override():
    yaml_content = """
    templates:
      base: &base
        backend: "Albumentations"
        transforms:
          - type: "Resize"
    
    transforms:
      template: *base
      backend: "Torchvision"
    """

    result = load_yaml(yaml_content)
    assert result["transforms"]["backend"] == "Torchvision"
    assert result["transforms"]["transforms"][0]["type"] == "Resize"
