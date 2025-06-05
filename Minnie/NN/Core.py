from ..LibImport import *
from ..Operation import *

# output_grad = ∂L/∂Y
# input_grad = ∂L/∂X = ∂L/∂Y * ∂Y/∂X
# param_grad = ∂L/∂W = ∂L/∂Y * ∂Y/∂W

class WeightsMultiplyOpt(ParamsOpt):        
    def __init__(self, W: ndarray, name: str = "weight_mutiply", **kwargs):
        super().__init__(W, name, **kwargs)
        self.in_features = W.shape[0]
        self.out_features = W.shape[1]
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return np.dot(inp, self.param)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))
    
    def compute_param_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(np.transpose(self.inp, (1, 0)), output_grad)
    

class BiasAddOpt(ParamsOpt):
    def __init__(self, bias, name: str = "bias_add", **kwargs):
        super().__init__(bias, name, **kwargs)
        # make sure bias.shape = (1, out_features)
        self.out_features = bias.shape[1]
        
    def compute_output(self):
        return self.inp + self.bias
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * np.ones(self.inp.shape)
    
    def compute_param_grad(self, output_grad: ndarray) -> ndarray:
        param_grad =  output_grad * np.ones(self.inp.shape)
        return np.expand_dims(np.sum(param_grad, axis = 0), axis = 0)
    
    