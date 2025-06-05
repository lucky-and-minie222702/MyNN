from ..LibImport import *
from .Operation import *
from . import Functional as F


class Sigmoid(Opt):
    def __init__(self):
        super().__init__("sigmoid")
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.sigmoid(inp)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.sigmoid_derivative(self.inp)
        return output_grad * grad
    

class Tanh(Opt):
    def __init__(self):
        super().__init__("tanh")
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.tanh(inp)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.sigmoid_derivative(self.inp)
        return output_grad * grad
    
    
class ReLU(Opt):
    def __init__(self):
        super().__init__("relu")
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.relu(inp)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.relu_derivative(self.inp)
        return output_grad * grad
    

class LeakyReLU(Opt):
    def __init__(self, alpha: float = 0.01):
        super().__init__("relu")
        self.alpha = alpha
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.leaky_relu(inp, self.alpha)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.leaky_relu_derivative(self.inp, self.alpha)
        return output_grad * grad
    
    
class SiLU(Opt):
    def __init__(self):
        super().__init__("silu")
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.silu(inp)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.silu_derivative(self.inp)
        return output_grad * grad
    
    
class GeLU(Opt):
    def __init__(self):
        super().__init__("gelu")
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.gelu(inp)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.gelu_derivative(self.inp)
        return output_grad * grad
    
    
class SeLU(Opt):
    def __init__(self):
        super().__init__("selu")
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.selu(inp)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.selu_derivative(self.inp)
        return output_grad * grad
    
    
class CeLU(Opt):
    def __init__(self):
        super().__init__("celu")
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.celu(inp)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.celu_derivative(self.inp)
        return output_grad * grad


class Softmax(Opt):
    def __init__(self, axis: int = -1):
        super().__init__("softmax")
        self.axis = axis
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.softmax(inp, self.axis)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.softmax_derivative(self.inp, self.axis)
        return output_grad * grad
    

class Softplus(Opt):
    def __init__(self):
        super().__init__("softplus")
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.softplus(inp)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.softplus_derivative(self.inp)
        return output_grad * grad


class Softsign(Opt):
    def __init__(self):
        super().__init__("softsign")
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return F.softsign(inp)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        grad = F.softsign_derivative(self.inp)
        return output_grad * grad


def activation_byname(name: str, **kwargs):
    available = ["sigmoid", "tanh", "relu", "leaky_relu", "silu", "gelu", "selu", "celu", "softmax", "softsign", "softplus"]
    name = name.lower()
    
    if name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "relu":
        return ReLU()
    elif name == "leaky_relu":
        return LeakyReLU(**kwargs)
    elif name == "silu":
        return SiLU()
    elif name == "gelu":
        return GeLU()
    elif name == "selu":
        return SeLU()
    elif name == "celu":
        return CeLU()
    elif name == "softmax":
        return Softmax(**kwargs)
    elif name == "softsign":
        return Softsign()
    elif name == "softplus":
        return Softplus()
    else:
        raise ValueError(f"Unknown built-in activation '{name}'\nAvailable built-in activations are: {" ".join(map(lambda x: f"'{x}'", available))}")