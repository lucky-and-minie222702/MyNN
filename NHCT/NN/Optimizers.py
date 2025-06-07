from ..Core import *
from .Operations import *
from . import Functional as F
from . import Modules


class Optimizer:
    def __init__(self, module: Modules.Module, lr: float):
        self.lr = lr
        self.module = module
        
    def collect(self):
        self.module.collect_layer_param_grads()
        self.module.collect_layer_params()
        
    def step(self):
        raise NotImplementedError()
        
    def zero_grads(self):
        for i in range(len(self.module.layer_params)): 
            for j in range(len(self.module.layer_params[i])):
                self.module.layer_param_grads[i][j] = 0
    
    def reset_after_epoch(self):
        pass
    
    
class SGD(Optimizer):
    def __init__(self, module: Modules.Module, lr: float = 0.01, momentum: float = 0.0):
        super(). __init__(module, lr)
        self.momentum = momentum
        self.velocities = None
        
    def zero_velocities(self):
        self.velocities = [[np.zeros(param.shape)
                            for param in params]
                                for params in self.module.layer_params]
        
    def step(self):
        self.collect()
        if self.velocities is None:
            self.velocities = [[np.zeros(param.shape)
                                for param in params]
                                    for params in self.module.layer_params]
            
        new_layer_params = []
            
        for i in range(len(self.module.layer_params)): 
            new_params = []
            for j in range(len(self.module.layer_params[i])):
                   self.velocities[i][j] *= self.momentum
                   self.velocities[i][j] += self.lr * self.module.layer_param_grads[i][j]
                   
                   new_params.append(self.module.layer_params[i][j] - self.velocities[i][j])
                   
            new_layer_params.append(new_params)
            
        self.module.set_layer_params(new_layer_params)
                   

def optimizer_byname(name: str, module: Modules.Module, **kwargs):
    name = name.lower()
    
    available = ["sgd"]
    name = name.lower()
    
    if name == "sgd":
        return SGD(module = module, **kwargs)
    else:
        _s = lambda x: f"'{x}'"
        raise ValueError(f"Unknown built-in optimizer '{name}'\nAvailable built-in optimizers are: {' '.join(map(_s, available))}")
    