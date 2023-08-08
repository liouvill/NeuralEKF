from typing import TYPE_CHECKING
import platform

if TYPE_CHECKING or platform.python_version().startswith("3.6."):
    from . import data, nn, utils
else:
    # Lazy submodule loading
    def __getattr__(name):
        import importlib

        module = importlib.import_module(__name__)
        if name not in __all__:
            raise AttributeError(f"{__name__!r} has no attribute {name!r}")
        imported = importlib.import_module(f".{name}", module.__spec__.parent)
        setattr(module, name, imported)
        return imported

__all__ = ["data", "nn", "utils"]
