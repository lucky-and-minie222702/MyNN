from ..LibImport import *
from .Operation import *
from . import Layer


class Module(NamedObj):
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.layer_params: List[ndarray] = []
        self.layer_param_grads: List[ndarray] = []
        
        self.input = None
        self.input_grad = None
        
        self.output = None
        self.output_grad = None
    
    def forward(self, inp: ndarray) -> ndarray:
        pass
    
    def backward(self, output_grad: ndarray) -> ndarray:
        pass

    def collect_layer_param_grads(self):
        pass
            
    def get_layer_param_grads(self) -> List[List[ndarray]]:
        pass
        
    def collect_layer_params(self):
        pass
            
    def get_layer_params(self) -> List[List[ndarray]]:
        pass
        
    def get_layer_names(self) -> List[str]:
        pass


class SequentialModule(Module):
    def __init__(self, layers: List[Layer.Layer]):
        super().__init__("sequential")
        self.layers = layers
        self.check_io_features()
        
        self.layer_params: List[ndarray] = []
        self.layer_param_grads: List[ndarray] = []
        
        self.input = None
        self.input_grad = None
        
        self.output = None
        self.output_grad = None
        
    def add(self, layer):
        self.layers.append(layer)
        self.check_io_features()
        
    def check_io_features(self):
        out_features = self.layers[0].out_features
        
        report = []
        
        for i, layer in enumerate(self.layers[1::], 1):
            if out_features is None or layer.in_features is None:
                continue
            
            if out_features != layer.in_features:
                report.append((
                    self.layers[i-1].get_config(),
                    self.layers[i].get_config()
                ))

            out_features = layer.out_features
            
        if len(report) > 0:
            raise ValueError(f"{"\n".join([f"Layers {l1} and {l2} are not compatible in terms features" for l1, l2 in report])}")
        
    def forward(self, inp: ndarray) -> ndarray:
        self.input = inp
        self.output = inp
        
        for layer in self.layers:
            self.output = layer.forward(self.output)
            
        return self.output
    
    def backward(self, output_grad: ndarray) -> ndarray:
        self.input_grad = output_grad
        
        for layer in reversed(self.layers):
            self.input_grad = layer.backward(self.input_grad)
            
        return self.input_grad

    def collect_layer_param_grads(self):
        self.layer_param_grads = []
        
        for layer in self.layers:
            self.layer_param_grads.append(layer.get_param_grads())
        
    def get_layer_param_grads(self, copy: bool = False, collect: bool = True) -> List[List[ndarray]]:
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
            
    def get_layer_params(self, copy: bool = False, collect: bool = True) -> List[List[ndarray]]:
        if collect:
            self.collect_layer_params()
        
        if copy:
            return self.layer_params.copy()
        else:
            return self.layer_params
        
    def get_layer_names(self) -> List[str]:
        return [layer.name for layer in self.layers]
    
    
class DynamicModule(Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass