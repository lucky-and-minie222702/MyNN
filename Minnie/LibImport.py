from typing import *
import numpy as np
from numpy import ndarray
import pandas as pd

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

class NamedObj:
    def __init__(self, name: str):
        self.name = name
    
    def get_config(self):
        return None