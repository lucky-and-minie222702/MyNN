from ..Core import *
from .Operations import *
from . import Functional as F
from .. import Math


class Loss(Opt):
    def __init__(self, name: str = ""):
        super().__init__(name)
        
        self.target = None
        self.prediction = None
        self.input_grad = None
        
        self.jit_compute_output = jit(self.compute_output)
        self.jit_compute_input_grad = jit(self.compute_input_grad)
        
    def forward(self, target: JArray, prediction: JArray, training: bool = True) -> JArray:
        self.target = target
        self.prediction = prediction
        
        loss = self.jit_compute_output(self.target, self.prediction, training = training)
        
        return loss
    
    def backward(self) -> JArray:
        self.input_grad = self.jit_compute_input_grad(self.target, self.prediction)
        
        assert self.input_grad.shape == self.prediction.shape
        
        return self.input_grad
        
    def compute_output(self, target: JArray, prediction: JArray) -> float:
        raise NotImplementedError()
    
    def compute_input_grad(self, target: JArray, prediction: JArray) -> JArray:
        return Math.derivative(lambda pred: self.compute_output(target, pred))
    
    
# CLASSIFICATION

class BinaryCrossentropy(Loss):
    def __init__(self, from_logits: bool = False):
        super().__init__("bce")
        self.from_logits = from_logits

    def compute_output(self, target: JArray, prediction: JArray, training: bool = True) -> float:
        return F.bce_loss(target, prediction, self.from_logits)
    
    def compute_input_grad(self, target: JArray, prediction: JArray) -> JArray:
        return F.bce_derivative(target, prediction, self.from_logits)
    

class CategoricalCrossentropy(Loss):
    def __init__(self, from_logits: bool = False, axis: int = -1):
        super().__init__("cce")
        self.from_logits = from_logits
        self.axis = axis

    def compute_output(self, target: JArray, prediction: JArray, training: bool = True) -> float:
        return F.cce_loss(target, prediction, self.from_logits, self.axis)
    
    def compute_input_grad(self, target: JArray, prediction: JArray) -> JArray:
        return F.cce_derivative(target, prediction, self.from_logits, self.axis)
    
    
class SparseCategoricalCrossentropy(Loss):
    def __init__(self, from_logits: bool = False, axis: int = -1):
        super().__init__("scce")
        self.from_logits = from_logits
        self.axis = axis

    def compute_output(self, target: JArray, prediction: JArray, training: bool = True) -> float:
        return F.scce_loss(target, prediction, self.from_logits, self.axis)
    
    def compute_input_grad(self, target: JArray, prediction: JArray) -> JArray:
        return F.scce_derivative(target, prediction, self.from_logits, self.axis)


# REGRESSION

class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__("mse")

    def compute_output(self, target: JArray, prediction: JArray, training: bool = True) -> float:
        return F.mse_loss(target, prediction)
    
    def compute_input_grad(self, target: JArray, prediction: JArray) -> JArray:
        return F.mse_derivative(target, prediction)
    
    
class RootMeanSquaredError(Loss):
    def __init__(self):
        super().__init__("mse")

    def compute_output(self, target: JArray, prediction: JArray, training: bool = True) -> float:
        return F.rmse_loss(target, prediction)
    
    def compute_input_grad(self, target: JArray, prediction: JArray) -> JArray:
        return F.rmse_derivative(target, prediction)
    

class MeanAbsoluteError(Loss):
    def __init__(self):
        super().__init__("mae")

    def compute_output(self, target: JArray, prediction: JArray, training: bool = True) -> float:
        return F.mae_loss(target, prediction)
    
    def compute_input_grad(self, target: JArray, prediction: JArray) -> JArray:
        return F.mae_derivative(target, prediction)
    
    
class MeanAbsolutePercentageError(Loss):
    def __init__(self):
        super().__init__("mape")

    def compute_output(self, target: JArray, prediction: JArray, training: bool = True) -> float:
        return F.mape_loss(target, prediction)
    
    def compute_input_grad(self, target: JArray, prediction: JArray) -> JArray:
        return F.mape_derivative(target, prediction)
    
    
class MeanSquaredLogarithmicError(Loss):
    def __init__(self):
        super().__init__("msle")

    def compute_output(self, target: JArray, prediction: JArray, training: bool = True) -> float:
        return F.msle_loss(target, prediction)
    
    def compute_input_grad(self, target: JArray, prediction: JArray) -> JArray:
        return F.msle_derivative(target, prediction)
    

def loss_byname(name: str, **kwargs):
    name = name.lower()
    
    available = ["mse", "rmse", "mae", "mape", "msle", "bce", "cce", "scce"]
    name = name.lower()
    
    if name == "mse":
        return MeanSquaredError(**kwargs)
    elif name == "rmse":
        return RootMeanSquaredError(**kwargs)
    elif name == "mae":
        return MeanAbsolutePercentageError(**kwargs)
    elif name == "mape":
        return MeanAbsolutePercentageError(**kwargs)
    elif name == "msle":
        return MeanSquaredLogarithmicError(**kwargs)
    elif name == "bce":
        return BinaryCrossentropy(**kwargs)
    elif name == "cce":
        return CategoricalCrossentropy(**kwargs)
    elif name == "scce":
        return SparseCategoricalCrossentropy(**kwargs)
    else:
        _s = lambda x: f"'{x}'"
        raise ValueError(f"Unknown built-in loss '{name}'\nAvailable built-in losses are: {' '.join(map(_s, available))}")