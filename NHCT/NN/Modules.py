from ..Core import *
from .Operations import *
from . import Layers
import pickle


class Module(NamedObj):
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.layer_params: List[List[JArray]] = []
        self.layer_param_grads: List[List[JArray]] = []
        
        self.input = None
        self.input_grad = None
        
        self.output = None
        self.output_grad = None
    
    def forward(self, inp: JArray, training: bool = True) -> JArray:
        raise NotImplementedError()
    
    def backward(self, output_grad: JArray) -> JArray:
        raise NotImplementedError()

    def collect_layer_param_grads(self):
        raise NotImplementedError()
        
    def get_layer_param_grads(self) -> List[List[JArray]]:
        raise NotImplementedError()
        
    def collect_layer_params(self):
        raise NotImplementedError()
            
    def get_layer_params(self) -> List[List[JArray]]:
        raise NotImplementedError()
        
    def get_layer_names(self) -> List[str]:
        raise NotImplementedError()
    
    def set_layer_params(self, new_layer_params: List[List[JArray]]):
        raise NotImplementedError()


class SequentialModule(Module):
    def __init__(self, layers: List[Layers.Layer]):
        super().__init__("sequential")
        self.layers = layers
        
        self.layer_params: List[List[JArray]] = []
        self.layer_param_grads: List[List[JArray]] = []
        
        self.input = None
        self.input_grad = None
        
        self.output = None
        self.output_grad = None
        
        self.check_subclass()
        
    def add(self, layer: Layers.Layer):
        assert issubclass(layer.__class__, Layers.Layer), "Layer must be a subclass of Layer"
        assert layer.in_features == self.layers[-1].out_features, f"Layers {self.layers[-1].get_config()} and {layer.get_config()} are not compatible in terms features"
        
        self.layers.append(layer)
        
    def build(self):
        for layer in self.layers:
            layer.build()
        
    def check_subclass(self):
        for layer in self.layers:
            assert issubclass(layer.__class__, Layers.Layer), "Layer must be a subclass of Layer"
        
    def forward(self, inp: JArray, training: bool = True, return_sequences: bool = False) -> JArray:
        self.input = inp
        self.output = inp
        sequences = []
        
        for i, layer in enumerate(self.layers):
            self.output = layer.forward(self.output, training = training)
            if return_sequences:
                sequences.append(self.output)
        
        if return_sequences:
            return sequences
        else:
            return self.output
    
    def backward(self, output_grad: JArray) -> JArray:
        self.input_grad = output_grad
        
        for layer in reversed(self.layers):
            self.input_grad = layer.backward(self.input_grad)
            
        return self.input_grad

    def collect_layer_param_grads(self):
        self.layer_param_grads = []
        
        for layer in self.layers:
            self.layer_param_grads.append(layer.get_param_grads())
        
    def get_layer_param_grads(self, copy: bool = False, collect: bool = True) -> List[List[JArray]]:
        if collect:
            self.collect_layer_param_grads()
        
        if copy:
            return self.layer_param_grads.copy()
        else:
            return self.layer_param_grads
        
    def collect_layer_params(self):
        self.layer_params = []
        
        for layer in self.layers:
            self.layer_params.append(layer.get_params())
            
    def set_layer_params(self, new_layer_params: List[List[JArray]]):
        for i, params in enumerate(new_layer_params):
            self.layers[i].set_params(params)
            
    def get_layer_params(self, copy: bool = False, collect: bool = True) -> List[List[JArray]]:
        if collect:
            self.collect_layer_params()
        
        if copy:
            return self.layer_params.copy()
        else:
            return self.layer_params
        
    def get_layer_names(self) -> List[str]:
        return [layer.name for layer in self.layers]
    
    def get_layer_opt_names(self, get_init_id: bool = True) -> List[str]:
        if get_init_id:
            return [(f"{layer.name}_{layer.init_id}", layer.get_opt_names(True)) for layer in self.layers]
        else:
            return [(f"{layer.name}", layer.get_opt_names()) for layer in self.layers]

    def save_pickle(self, file: str, collect: bool = True):
        if collect:
            self.collect_layer_params()
            
        with open(file, "wb") as f:
            pickle.dump(self.layer_params, f)
            
    def load_pickle(self, file: str):
        with open(file, "rb") as f:
            self.set_layer_params(pickle.load(f))
    
    
class DynamicModule(Module):
    def __init__(self):
        pass