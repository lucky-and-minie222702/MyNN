from ..Core import *
from .Operations import *
from . import Functional as F

class LRSchdule(NamedObj):
    def __init__(self, lr: float, name: str = ""):
        super().__init__(name)
        self.lr = lr
        
    def __call__(self, epoch: int, current_lr: float, **kwargs):
        self.lr = self.update()
        
    def update(self, epoch: int, current_lr: float, **kwargs) -> float:
        raise NotImplementedError()
        
