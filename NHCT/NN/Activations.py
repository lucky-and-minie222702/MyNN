from ..Core import *
from .Operations import *
from . import Functional as F


class Sigmoid(Opt):
    def __init__(self):
        super().__init__("sigmoid")
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.sigmoid(inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.sigmoid_derivative(inp)
        return output_grad * grad
    

class Tanh(Opt):
    def __init__(self):
        super().__init__("tanh")
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.tanh(inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.sigmoid_derivative(inp)
        return output_grad * grad
    
    
class ReLU(Opt):
    def __init__(self):
        super().__init__("relu")
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.relu(inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.relu_derivative(inp)
        return output_grad * grad
    

class LeakyReLU(Opt):
    def __init__(self, alpha: float = 0.01):
        super().__init__("relu")
        self.alpha = alpha
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.leaky_relu(inp, self.alpha)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.leaky_relu_derivative(inp, self.alpha)
        return output_grad * grad
    
    
class SiLU(Opt):
    def __init__(self):
        super().__init__("silu")
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.silu(inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.silu_derivative(inp)
        return output_grad * grad
    
    
class GeLU(Opt):
    def __init__(self):
        super().__init__("gelu")
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.gelu(inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.gelu_derivative(inp)
        return output_grad * grad
    
    
class SeLU(Opt):
    def __init__(self):
        super().__init__("selu")
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.selu(inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.selu_derivative(inp)
        return output_grad * grad
    
    
class CeLU(Opt):
    def __init__(self):
        super().__init__("celu")
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.celu(inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.celu_derivative(inp)
        return output_grad * grad


class Softmax(Opt):
    def __init__(self, axis: int = -1):
        super().__init__("softmax")
        self.axis = axis
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.softmax(inp, self.axis)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.softmax_derivative(inp, self.axis)
        return output_grad * grad


class Softplus(Opt):
    def __init__(self):
        super().__init__("softplus")
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.softplus(inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.softplus_derivative(inp)
        return output_grad * grad


class Softsign(Opt):
    def __init__(self):
        super().__init__("softsign")
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return F.softsign(inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        grad = F.softsign_derivative(inp)
        return output_grad * grad


def activation_byname(name: str, **kwargs):
    available = ["sigmoid", "tanh", "relu", "leaky_relu", "silu", "gelu", "selu", "celu", "softmax", "softsign", "softplus"]
    name = name.lower()
    
    if name == "sigmoid":
        return Sigmoid(**kwargs)
    elif name == "tanh":
        return Tanh(**kwargs)
    elif name == "relu":
        return ReLU(**kwargs)
    elif name == "leaky_relu":
        return LeakyReLU(**kwargs)
    elif name == "silu":
        return SiLU(**kwargs)
    elif name == "gelu":
        return GeLU(**kwargs)
    elif name == "selu":
        return SeLU(**kwargs)
    elif name == "celu":
        return CeLU(**kwargs)
    elif name == "softmax":
        return Softmax(**kwargs)
    elif name == "softsign":
        return Softsign(**kwargs)
    elif name == "softplus":
        return Softplus(**kwargs)
    else:
        _s = lambda x: f"'{x}'"
        raise ValueError(f"Unknown built-in activation '{name}'\nAvailable built-in activations are: {' '.join(map(_s, available))}")