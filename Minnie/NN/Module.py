from ..LibImport import *
from .Operation import *
from . import Functional as F


class Module:
    def __init__(self):
        self.layer_params: List[ndarray] = []
    
    def forward(self):
        pass
    
    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
