from Minnie.NN import Initializer
from ..LibImport import *
from .Operation import *
from .Activation import *
from .Initializer import *
from . import Functional as F


_LAYER_COUNT = 0


################
#              #
#  Core LAYER  #
#              #
################

class Layer(NamedObj):
    def __init__(self, name: str = ""):
        super().__init__(name)
        
        self.initialized = False
        self.operations: List[Opt] = []
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []

        self.output = None
        self.output_grad = None
        
        self.input = None
        self.input_grad = None
        
        self.in_features = None
        self.out_features = None
        
    def build(self):
        raise NotImplementedError()
    
    def forward(self, inp: ndarray) -> ndarray:
        if not self.initialized:
            self.build()
            self.initialized = True
            
        self.input = inp
        self.output = inp
         
        for opt in self.operations:
            self.output = opt.forward(self.output)
        
        return self.output
    
    def backward(self, output_grad: ndarray) -> ndarray:
        assert self.output.shape == output_grad.shape
        
        self.input_grad = output_grad
        for opt in reversed(self.operations):
            self.input_grad = opt.backward(self.input_grad)
            
        return self.input_grad
    
    def collect_param_grads(self):
        self.param_grads = []
        
        for opt in self.operations:
            if issubclass(opt.__class__, ParamOpt):
                self.param_grads.append(opt.param_grad)

    def get_param_grads(self, copy: bool = False, collect: bool = True) -> List[ndarray]:
        if collect:
            self.collect_param_grads()
        
        if copy:
            return self.param_grads.copy()
        else:
            return self.param_grads
                
    def collect_params(self):
        self.params = []
        
        for opt in self.operations:
            if issubclass(opt.__class__, ParamOpt):
                self.params.append(opt.param)
            
    def get_params(self, copy: bool = False, collect: bool = True) -> List[ndarray]:
        if collect:
            self.collect_params()
        
        if copy:
            return self.params.copy()
        else:
            return self.params
                
    def get_opt_names(self) -> List[str]:
        return [opt.name for opt in self.operations]
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }
    
    
###########
#         #
#  LAYER  #
#         #
###########

class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, activation: str | Opt | None = None, initializer: str | Callable = "xavier", bias: bool = True, name: str = None):
        global _LAYER_COUNT
        _LAYER_COUNT += 1

        if name is None:
            name = f"dense_{_LAYER_COUNT}"
            
        super().__init__(name)
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.bias = bias
        
        self.activation = activation
        self.initializer = initializer
        
        if isinstance(self.activation, str):
            self.activation = activation_byname(self.activation)
            
        if isinstance(self.initializer, str):
            self.initializer = initializer_byname(self.initializer)
        
        
    def build(self):
        self.operations.append(
            WeightMultiplyOpt(self.initializer(self.in_features, self.out_features))
        )
        
        if self.bias:
            self.operations.append(
                BiasAddOpt(self.initializer(1, self.out_features))
            )

        if self.activation is not None:
            self.operations.append(
                self.activation
            )



class Activation(Layer):
    def __init__(self, activation_name: str = None, name: str = None):
        global _LAYER_COUNT
        _LAYER_COUNT += 1

        if name is None:
            name = f"{'activation' if activation_name is None else activation_name}_{_LAYER_COUNT}"
            
        super().__init__(name)
        
        self.f = activation_name
        
        if isinstance(self.f, str):
            self.f = activation_byname(self.f)
        
    def build(self):
        self.operations.append(
            self.f
        )
