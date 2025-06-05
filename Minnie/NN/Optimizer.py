from ..LibImport import *
from .Operation import *
from . import Functional as F
from . import Module


class Optimizer:
    def __init__(self, lr: float):
        self.lr = lr
        self.module = None
        
    def assign(self, module: Module.Module):
        self.module = module
        
    def step(self):
        raise NotImplementedError()
    
    
class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        super(). __init__(lr)
        
    def step(self):
        self.module.collect_layer_param_grads()
        self.module.collect_layer_params()
        for layer_pair in zip(self.module.layer_params, self.module.layer_param_grads):
            for param, param_grad in layer_pair:
                param -= self.lr * param_grad