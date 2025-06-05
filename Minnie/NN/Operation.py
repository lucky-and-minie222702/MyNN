from ..LibImport import *


# input_grad  : gradient respects to input of the operation
# output_grad : gradient respects to output of the operation

class Opt(NamedObj):
    def __init__(self, name: str = ""):
        super().__init__(name)
        
        self.input = None
        self.input_grad = None
        
        self.output = None
        self.output_grad = None
                
    def forward(self, inp: ndarray):
        self.input = inp
        self.output = self.compute_output(self.input)
        return self.output
        
    def backward(self, output_grad: ndarray) -> ndarray:
        self.output_grad = output_grad
        self.input_grad = self.compute_input_grad(self.output_grad)
        
        assert self.input.shape == self.input_grad.shape
        assert self.output.shape == self.output_grad.shape
        
        return self.input_grad
    
    def compute_output(self, inp: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()
    

class ParamOpt(Opt):
    def __init__(self, param: ndarray, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.param = param
        self.param_grad = None
        
    def backward(self, output_grad: ndarray) -> ndarray:
        self.output_grad = output_grad
        self.input_grad = self.compute_input_grad(self.output_grad)
        self.param_grad = self.compute_param_grad(self.output_grad)
        
        assert self.input.shape == self.input_grad.shape
        assert self.output.shape == self.output_grad.shape
        assert self.param.shape == self.param_grad.shape
        
        return self.input_grad
    
    def compute_param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()


###########
#         #
#  Basic  #
#         #
###########

# output_grad = ∂L/∂Y
# input_grad = ∂L/∂X = ∂L/∂Y * ∂Y/∂X
# param_grad = ∂L/∂W = ∂L/∂Y * ∂Y/∂W

class WeightMultiplyOpt(ParamOpt):        
    def __init__(self, W: ndarray, name: str = "weight_mutiply", **kwargs):
        super().__init__(W, name, **kwargs)
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return np.dot(inp, self.param)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))
    
    def compute_param_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(np.transpose(self.input, (1, 0)), output_grad)
    

class BiasAddOpt(ParamOpt):
    def __init__(self, bias, name: str = "bias_add", **kwargs):
        super().__init__(bias, name, **kwargs)
        # make sure bias.shape = (1, out_features)
        assert self.param.shape[0] == 1
        
    def compute_output(self, inp: ndarray) -> ndarray:
        return inp + self.param
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * np.ones(self.input.shape)
    
    def compute_param_grad(self, output_grad: ndarray) -> ndarray:
        param_grad = output_grad * np.ones(self.input.shape)
        return np.expand_dims(np.sum(param_grad, axis = 0), axis = 0)
