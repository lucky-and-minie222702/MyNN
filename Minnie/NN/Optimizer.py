from ..LibImport import *
from .Operation import *
from . import Functional as F


class Optimizer:
    def __init__(self, params: ndarray, lr: float):
        self.lr = lr
        self.params = params
        
    def step(self):
        raise NotImplementedError()
    
    
# class SGD(Optimizer):
#     def __init__(self, lr: float = 0.01):
#         super(). __init__(lr)
        
#     def step(self):
#         for param, param_grad in zip()