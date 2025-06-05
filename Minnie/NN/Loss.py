from ..LibImport import *
from .Operation import *
from . import Functional as F


class Loss(Opt):
    def __init__(self, name: str = ""):
        super().__init__(name)
        
        self.target = None
        self.prediction = None
        self.input_grad = None
        
    def forward(self, target: ndarray, prediction: ndarray) -> ndarray:
        self.target = target
        self.prediction = prediction
        
        loss = self.compute_output(self.target, self.prediction)
        
        return loss
    
    def backward(self) -> ndarray:
        self.input_grad = self.compute_input_grad(self.target, self.prediction)
        
        assert self.input_grad.shape == self.prediction.shape
        
        return self.input_grad
        
    def compute_output(self, target: ndarray, prediction: ndarray) -> float:
        raise NotImplementedError()
    
    def compute_input_grad(self, target: ndarray, prediction: ndarray) -> ndarray:
        raise NotImplementedError()
    
    
# CLASSIFICATION

class BinaryCrossentropy(Loss):
    def __init__(self, from_logits: bool = False):
        super().__init__("bce")
        self.from_logits = from_logits

    def compute_output(self, target: ndarray, prediction: ndarray) -> float:
        return F.bce(target, prediction, self.from_logits)
    
    def compute_input_grad(self, target: ndarray, prediction: ndarray) -> ndarray:
        return F.bce_derivative(target, prediction, self.from_logits)
    

class CategoricalCrossentropy(Loss):
    def __init__(self, from_logits: bool = False, axis: int = -1):
        super().__init__("cce")
        self.from_logits = from_logits
        self.axis = axis

    def compute_output(self, target: ndarray, prediction: ndarray) -> float:
        return F.cce(target, prediction, self.from_logits, self.axis)
    
    def compute_input_grad(self, target: ndarray, prediction: ndarray) -> ndarray:
        return F.cce_derivative(target, prediction, self.from_logits, self.axis)
    
    
class SparseCategoricalCrossentropy(Loss):
    def __init__(self, from_logits: bool = False, axis: int = -1):
        super().__init__("scce")
        self.from_logits = from_logits
        self.axis = axis

    def compute_output(self, target: ndarray, prediction: ndarray) -> float:
        return F.scce(target, prediction, self.from_logits, self.axis)
    
    def compute_input_grad(self, target: ndarray, prediction: ndarray) -> ndarray:
        return F.scce_derivative(target, prediction, self.from_logits, self.axis)


# REGRESSION

class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__("mse")

    def compute_output(self, target: ndarray, prediction: ndarray) -> float:
        return F.mse(target, prediction)
    
    def compute_input_grad(self, target: ndarray, prediction: ndarray) -> ndarray:
        return F.mse_derivative(target, prediction)
    
    
class RootMeanSquaredError(Loss):
    def __init__(self):
        super().__init__("mse")

    def compute_output(self, target: ndarray, prediction: ndarray) -> float:
        return F.rmse(target, prediction)
    
    def compute_input_grad(self, target: ndarray, prediction: ndarray) -> ndarray:
        return F.rmse_derivative(target, prediction)
    

class MeanAbsoluteError(Loss):
    def __init__(self):
        super().__init__("mae")

    def compute_output(self, target: ndarray, prediction: ndarray) -> float:
        return F.mae(target, prediction)
    
    def compute_input_grad(self, target: ndarray, prediction: ndarray) -> ndarray:
        return F.mae_derivative(target, prediction)
    
    
class MeanAbsolutePercentageError(Loss):
    def __init__(self):
        super().__init__("mape")

    def compute_output(self, target: ndarray, prediction: ndarray) -> float:
        return F.mape(target, prediction)
    
    def compute_input_grad(self, target: ndarray, prediction: ndarray) -> ndarray:
        return F.mape_derivative(target, prediction)
    
    
class MeanSquaredLogarithmicError(Loss):
    def __init__(self):
        super().__init__("msle")

    def compute_output(self, target: ndarray, prediction: ndarray) -> float:
        return F.msle(target, prediction)
    
    def compute_input_grad(self, target: ndarray, prediction: ndarray) -> ndarray:
        return F.msle_derivative(target, prediction)
    

def loss_byname(name: str, **kwargs):
    name = name.lower()
    
    available = ["mse", "rmse", "mae", "mape", "msle", "bce", "cce", "scce"]
    name = name.lower()
    
    if name == "mse":
        return MeanSquaredError()
    elif name == "rmse":
        return RootMeanSquaredError()
    elif name == "mae":
        return MeanAbsolutePercentageError()
    elif name == "mape":
        return MeanAbsolutePercentageError()
    elif name == "msle":
        return MeanSquaredLogarithmicError()
    elif name == "bce":
        return BinaryCrossentropy(**kwargs)
    elif name == "cce":
        return CategoricalCrossentropy(**kwargs)
    elif name == "scce":
        return SparseCategoricalCrossentropy(**kwargs)
    else:
        _s = lambda x: f"'{x}'"
        raise ValueError(f"Unknown built-in loss '{name}'\nAvailable built-in losses are: {' '.join(map(_s, available))}")