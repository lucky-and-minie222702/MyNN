from ..Core import *
from .Operations import *
from . import Functional as F

# epoch start from 1

class LRSchedule(NamedObj):
    def __init__(self, lr: float, name: str = ""):
        super().__init__(name)
        self.lr = lr
        
    def __call__(self, epoch: int, lr: float = None, **kwargs):
        self.lr = self.update(epoch, lr if lr is not None else self.lr, **kwargs)
        return self.lr
        
    def update(self, epoch: int, lr: float = None, **kwargs) -> float:
        raise NotImplementedError


class LRLinearDecay(LRSchedule):
    def __init__(self, start_lr: float, end_lr: float, epochs: int, name: str = ""):
        super().__init__(start_lr, name)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        
    def update(self, epoch: int, lr: float = None, **kwargs) -> float:
        return (self.start_lr - self.end_lr) * epoch / self.epochs


class LRExponentialDecay(LRSchedule):
    def __init__(self, start_lr: float, end_lr: float, epochs: int, name: str = ""):
        super().__init__(start_lr, name)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        
    def update(self, epoch: int, lr: float = None, **kwargs) -> float:
        return (self.start_lr - self.end_lr) ** (1 / (self.epochs - 1))