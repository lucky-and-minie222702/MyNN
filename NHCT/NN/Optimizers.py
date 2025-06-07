from ..Core import *
from .Operations import *
from . import Functional as F
from . import Modules


class Optimizer:
    def __init__(self, module: Modules.Module, lr: float):
        self.lr = lr
        self.module = module
        
    def step(self):
        self.module.collect_layer_param_grads()
        self.module.collect_layer_params()
    
    
class SGD(Optimizer):
    def __init__(self, module: Modules.Module, lr: float = 0.01):
        super(). __init__(module, lr)
        
    def step(self):
        super().step()
        for i in range(len(self.module.layer_params)): 
            for j in range(len(self.module.layer_params[i])):
                   self.module.layer_params[i][j] -= self.lr * self.module.layer_param_grads[i][j]
                   

def optimizer_byname(name: str, module: Modules.Module, **kwargs):
    name = name.lower()
    
    available = ["sgd"]
    name = name.lower()
    
    if name == "sgd":
        return SGD(module = module, **kwargs)
    else:
        _s = lambda x: f"'{x}'"
        raise ValueError(f"Unknown built-in optimizer '{name}'\nAvailable built-in optimizers are: {' '.join(map(_s, available))}")
    