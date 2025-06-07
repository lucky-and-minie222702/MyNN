from typing import *
import pandas as pd

# jax
import jax
import jax.numpy as jnp
from jax import Array as JArray
from jax import jit, lax

# numpy
import numpy as np
from numpy import ndarray

OBJ_COUNT = 0
EPSILON = 1e-8
PRINT_OBJ_INIT_LOG = False
INIT_LOG = []

class NamedObj:
    def __init__(self, name: str):
        self.name = name
        
        global OBJ_COUNT
        global PRINT_OBJ_INIT_LOG
        global INIT_LOG
        
        self.init_log = f"Object [{OBJ_COUNT}] |name: {name} | class: {self.__class__.__name__}"
        INIT_LOG.append(self.init_log)
        if PRINT_OBJ_INIT_LOG:
            print(self.init_log)
            
        OBJ_COUNT += 1
        self.init_id = OBJ_COUNT
    
    def get_config(self):
        return None