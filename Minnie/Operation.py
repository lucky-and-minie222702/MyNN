from LibImport import *

class Opt(NamedObj):
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.inp = None
        self.out_ = None
        self.input_grad = None
        self.output_grad = None
        
        self.in_features = None
        self.out_features = None
                
    def forward(self, inp: ndarray):
        self.inp = inp
        self.out_ = self.compute_output(self.inp)
        return self.out_
        
    def backward(self, output_grad: ndarray) -> ndarray:
        self.output_grad = output_grad
        self.input_grad = self.compute_input_grad(self.output_grad)
        return self.input_grad
    
    def compute_output(self, inp: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def compute_input_grad(self, output_grad: ndarray):
        raise NotImplementedError()
    

class ParamsOpt(Opt):
    def __init__(self, param: ndarray, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.param = param
        self.param_grad = None
        
    def backward(self, output_grad: ndarray) -> ndarray:
        self.output_grad = output_grad
        self.input_grad = self.compute_input_grad(self.output_grad)
        self.param_grad = self.compute_param_grad(self.output_grad)
        return self.input_grad
    
    def compute_param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()
