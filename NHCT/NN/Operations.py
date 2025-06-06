from ..Core import *


# input_grad  : gradient respects to input of the operation
# output_grad : gradient respects to output of the operation

class Opt(NamedObj):
    def __init__(self, name: str = ""):
        super().__init__(name)
        
        self.input = None
        self.input_grad = None
        
        self.output = None
        self.output_grad = None
                
    def forward(self, inp: ndarray, training: bool = True) -> ndarray:
        self.input = inp
        self.output = self.compute_output(self.input, training = training)
        return self.output
        
    def backward(self, output_grad: ndarray) -> ndarray:
        self.output_grad = output_grad
        self.input_grad = self.compute_input_grad(self.output_grad)
        
        assert self.input.shape == self.input_grad.shape
        assert self.output.shape == self.output_grad.shape
        
        return self.input_grad
    
    def compute_output(self, inp: ndarray, training: bool = True) -> ndarray:
        raise NotImplementedError()
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()
    

class ParamOpt(Opt):
    def __init__(self, param: ndarray, name: str):
        super().__init__(name)
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
    
    def set_param(self, new_param: ndarray):
        assert self.param.shape == new_param.shape
        self.param = new_param
    
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
    def __init__(self, W: ndarray, name: str = "weight_mutiply"):
        super().__init__(W, name)
        
        # param: (in_features, out_featrues)
        # input: (batch_size, in_features)
        # output: (batch_size, out_features)
        
    def compute_output(self, inp: ndarray, training: bool = True) -> ndarray:
        return np.dot(inp, self.param)
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))
    
    def compute_param_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(np.transpose(self.input, (1, 0)), output_grad)
    

class BiasAddOpt(ParamOpt):
    def __init__(self, bias, name: str = "bias_add"):
        super().__init__(bias, name)
        # make sure bias.shape = (1, out_features)
        assert self.param.shape[0] == 1
        
    def compute_output(self, inp: ndarray, training: bool = True) -> ndarray:
        return inp + self.param
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * np.ones(self.input.shape)
    
    def compute_param_grad(self, output_grad: ndarray) -> ndarray:
        param_grad = output_grad * np.ones(self.input.shape)
        return np.expand_dims(np.sum(param_grad, axis = 0), axis = 0)
    
    
class DropoutOpt(Opt):
    def __init__(self, rate: float = 0.0, name: str = "dropout"):
        super().__init__(name)
        
        self.mask = None
        
        self.rate = rate
        assert 0.0 <= self.rate <= 1.0, f"Invalid rate {rate}"
        
    def compute_output(self, inp: ndarray, training: bool = True) -> ndarray:
        self.mask = np.random.rand(*inp.shape) > self.rate
        if training:
            return self.mask * inp
        else:
            return inp
    
    def compute_input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * self.mask