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
        
        self.jit_compute_output = jit(self.compute_output)
        self.jit_compute_input_grad = jit(self.compute_input_grad)
                
    def forward(self, inp: JArray, training: bool = True) -> JArray:
        self.input = inp
        self.output = self.jit_compute_output(self.input, training = training)
        return self.output
        
    def backward(self, output_grad: JArray) -> JArray:
        self.output_grad = output_grad
        self.input_grad = self.jit_compute_input_grad(self.input, self.output_grad)
        
        assert self.input.shape == self.input_grad.shape
        assert self.output.shape == self.output_grad.shape
        
        return self.input_grad
    
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        raise NotImplementedError()
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        raise NotImplementedError()
    

class ParamOpt(Opt):
    def __init__(self, param: JArray, name: str):
        super().__init__(name)
        self.param = param
        self.param_grad = None
        self.jit_compute_param_grad = jit(self.compute_param_grad)
        
    def forward(self, inp: JArray, training: bool = True) -> JArray:
        self.input = inp
        self.output = self.jit_compute_output(self.param, self.input, training = training)
        return self.output
        
    def backward(self, output_grad: JArray) -> JArray:
        self.output_grad = output_grad
        self.input_grad = self.jit_compute_input_grad(self.param, self.input, self.output_grad)
        self.param_grad = self.jit_compute_param_grad(self.param, self.input, self.output_grad)
        
        assert self.input.shape == self.input_grad.shape
        assert self.output.shape == self.output_grad.shape
        assert self.param.shape == self.param_grad.shape
        
        return self.input_grad
    
    def set_param(self, new_param: JArray):
        assert self.param.shape == new_param.shape
        self.param = new_param
        exit()
        
    def compute_output(self, param: JArray, inp: JArray, training: bool = True) -> JArray:
        raise NotImplementedError()
    
    def compute_input_grad(self, param: JArray, inp: JArray, output_grad: JArray) -> JArray:
        raise NotImplementedError()
    
    def compute_param_grad(self, param: JArray, inp: JArray, output_grad: JArray) -> JArray:
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
    def __init__(self, W: JArray, name: str = "weight_mutiply"):
        super().__init__(W, name)
        
        # param: (in_features, out_featrues)
        # input: (batch_size, in_features)
        # output: (batch_size, out_features)

    def compute_output(self, param: JArray, inp: JArray, training: bool = True) -> JArray:
        return jnp.dot(inp, param)
    
    def compute_input_grad(self, param: JArray, inp: JArray, output_grad: JArray) -> JArray:
        return jnp.dot(output_grad, jnp.transpose(param, (1, 0)))
    
    def compute_param_grad(self, param: JArray, inp: JArray, output_grad: JArray) -> JArray:
        return jnp.dot(jnp.transpose(inp, (1, 0)), output_grad)
    

class BiasAddOpt(ParamOpt):
    def __init__(self, bias, name: str = "bias_add"):
        super().__init__(bias, name)
        # make sure bias.shape = (1, out_features)
        assert self.param.shape[0] == 1
        
    def compute_output(self, param: JArray, inp: JArray, training: bool = True) -> JArray:
        return inp + param
    
    def compute_input_grad(self, param: JArray, inp: JArray, output_grad: JArray) -> JArray:
        return output_grad * jnp.ones(inp.shape)
    
    def compute_param_grad(self, param: JArray, inp: JArray, output_grad: JArray) -> JArray:
        param_grad = output_grad * jnp.ones(inp.shape)
        return jnp.expand_dims(jnp.sum(param_grad, axis = 0), axis = 0)
    
    
class DropoutOpt(Opt):
    def __init__(self, rate: float = 0.0, name: str = "dropout"):
        super().__init__(name)
        
        self.mask = None
        
        self.rate = rate
        assert 0.0 <= self.rate <= 1.0, f"Invalid rate {rate}"
        
    def forward(self, inp: JArray, training: bool = True) -> JArray:
        self.input = inp
        
        self.mask = np.random.rand(*inp.shape) > self.rate
        self.mask = jnp.asarray(self.mask)
        
        self.output = self.jit_compute_output(self.input, training = training)
        return self.output
        
    def compute_output(self, inp: JArray, training: bool = True) -> JArray:
        return lax.cond(training, lambda: self.mask * inp, lambda: inp)
    
    def compute_input_grad(self, inp: JArray, output_grad: JArray) -> JArray:
        return output_grad * self.mask