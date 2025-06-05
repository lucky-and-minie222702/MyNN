from .LibImport import *

def deriv(func: Array_Function, inp: ndarray, delta: float = 1e-6):
    return (func(inp + delta) - func(inp - delta)) / (2 * delta)