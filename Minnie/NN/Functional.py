from ..LibImport import *
from scipy.special import erf

################
#              #
#  ACTIVATION  #
#              #
################


# Sigmoid
def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: ndarray) -> ndarray:
    s = sigmoid(x)
    return s * (1 - s)

# Tanh
def tanh(x: ndarray) -> ndarray:
    return np.tanh(x)

def tanh_derivative(x: ndarray) -> ndarray:
    return 1 - np.tanh(x) ** 2


# ReLU
def relu(x: ndarray) -> ndarray:
    return np.maximum(0, x)

def relu_derivative(x: ndarray) -> ndarray:
    return (x > 0).astype(float)


# Leaky ReLU
def leaky_relu(x: ndarray, alpha: float) -> ndarray:
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: ndarray, alpha: float) -> ndarray:
    return np.where(x > 0, 1.0, alpha)


# GeLU
def gelu(x: ndarray) -> ndarray:
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_derivative(x: ndarray) -> ndarray:
    phi = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return 0.5 * (1 + erf(x / np.sqrt(2))) + 0.5 * x * phi


# SiLU (Swish)
def silu(x: ndarray) -> ndarray:
    return x * sigmoid(x)

def silu_derivative(x: ndarray) -> ndarray:
    s = sigmoid(x)
    return s * (1 + x * (1 - s))


# Softplus
def softplus(x: ndarray) -> ndarray:
    return np.log1p(np.exp(x))

def softplus_derivative(x: ndarray) -> ndarray:
    return sigmoid(x)


# Softsign
def softsign(x: ndarray) -> ndarray:
    return x / (1 + np.abs(x))

def softsign_derivative(x: ndarray) -> ndarray:
    return 1 / (1 + np.abs(x))**2


# Softmax
def softmax(x: ndarray) -> ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_derivative(x: ndarray) -> ndarray:
    s = softmax(x).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


# SeLU
SELU_ALPHA = 1.6732632423543772
SELU_LAMBDA = 1.0507009873554805

def selu(x: ndarray) -> ndarray:
    return SELU_LAMBDA * np.where(x > 0, x, SELU_ALPHA * (np.exp(x) - 1))

def selu_derivative(x: ndarray) -> ndarray:
    return SELU_LAMBDA * np.where(x > 0, 1.0, SELU_ALPHA * np.exp(x))


# CeLU
def celu(x: ndarray, alpha: float = 1.0) -> ndarray:
    return np.where(x >= 0, x, alpha * (np.exp(x / alpha) - 1))

def celu_derivative(x: ndarray, alpha: float = 1.0) -> ndarray:
    return np.where(x >= 0, 1.0, np.exp(x / alpha))