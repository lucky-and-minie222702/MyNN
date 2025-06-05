from ..LibImport import *
from .Operation import *
from . import Functional as F

class He:
    def __init__(self):
        pass

    def __call__(self, *shape):
        return F.he_init(*shape)

    
class Xavier:
    def __init__(self):
        pass

    def __call__(self, *shape):
        return F.xavier_init(*shape)
    

class LeCun:
    def __init__(self):
        pass

    def __call__(self, *shape):
        return F.lecun_init(*shape)

    
class Normal01:
    def __init__(self):
        pass

    def __call__(self, *shape):
        return F.normal01_init(*shape)
    

class Normal:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, *shape):
        return F.normal_init(*shape, mean = self.mean, std = self.std)
    

class Uniform:
    def __init__(self, low: float = -1.0, high: float = 1.0):
        self.low = low
        self.high = high

    def __call__(self, *shape):
        return F.uniform_init(*shape, low = self.low, high = self.high)
    

def initializer_byname(name: str, **kwargs):
    available = ["he", "xavier", "lecun", "normal01", "normal", "uniform"]
    name = name.lower()
    
    if name == "he":
        return He()
    elif name == "xavier":
        return Xavier()
    elif name == "lecun":
        return LeCun()
    elif name == "normal01":
        return Normal01()
    elif name == "normal":
        return Normal(**kwargs)
    elif name == "uniform":
        return Uniform(**kwargs)
    else:
        raise ValueError(f"Unknown built-in initializer '{name}'\nAvailable built-in initializers are: {" ".join(map(lambda x: f"'{x}'", available))}")