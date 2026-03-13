from src.modeling.augmentation import Augmenter
from src.modeling.context_assembler import ContextAssembler

def __getattr__(name):
    if name == "ByT5Trainer":
        from src.modeling.byt5_trainer import ByT5Trainer
        return ByT5Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Augmenter", "ContextAssembler", "ByT5Trainer"]
