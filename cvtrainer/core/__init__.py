from .config import load_yaml, load_yaml_file

__all__ = ["load_yaml", "load_yaml_file", "Stage"]


def __getattr__(name):
    if name == "Stage":
        from .stage import Stage

        return Stage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
