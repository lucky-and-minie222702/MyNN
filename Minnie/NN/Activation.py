from ..LibImport import *
from ..Operation import *
import Functional as F

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
    def __init__(self, alpha):
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