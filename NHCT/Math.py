from .Core import *

def derivative(func: Callable[[ndarray], ndarray], inp: ndarray, delta: float = 1e-6):
    return (func(inp + delta) - func(inp - delta)) / (2 * delta)