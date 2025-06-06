from ..Core import *
from .Operations import *
from . import Functional as F


class Initializer(NamedObj):
    pass

class He(Initializer):
    def __init__(self):
        super().__init__("he")
        pass

    def __call__(self, *shape):
        return F.he_init(*shape)

    
class Xavier(Initializer):
    def __init__(self):
        super().__init__("xavier")
        pass

    def __call__(self, *shape):
        return F.xavier_init(*shape)
    

class LeCun(Initializer):
    def __init__(self):
        super().__init__("lecun")
        pass

    def __call__(self, *shape):
        return F.lecun_init(*shape)


class Normal(Initializer):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__("normal")
        self.mean = mean
        self.std = std

    def __call__(self, *shape):
        return F.normal_init(*shape, mean = self.mean, std = self.std)
    

class Uniform(Initializer):
    def __init__(self, low: float = -1.0, high: float = 1.0):
        super().__init__("uniform")
        self.low = low
        self.high = high

    def __call__(self, *shape):
        return F.uniform_init(*shape, low = self.low, high = self.high)
    

def initializer_byname(name: str, **kwargs):
    available = ["he", "xavier", "lecun", "normal", "uniform"]
    name = name.lower()
    
    if name == "he":
        return He()
    elif name == "xavier":
        return Xavier()
    elif name == "lecun":
        return LeCun()
    elif name == "normal":
        return Normal(**kwargs)
    elif name == "uniform":
        return Uniform(**kwargs)
    else:
        _s = lambda x: f"'{x}'"
        raise ValueError(f"Unknown built-in initializer '{name}'\nAvailable built-in initializers are: {' '.join(map(_s, available))}")