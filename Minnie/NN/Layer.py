from ..LibImport import *
from ..Operation import *

class Layer(NamedObj):
    def __init__(self, name: str = ""):
        super().__init__(name)
        
        self.params = []
        self.param_grads = []
        self.operations = []
        
        