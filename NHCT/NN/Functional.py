from ..Core import *
from scipy.special import erf

#################
#               #
#  Initializer  #
#               #
#################


def normal_init(*shape, mean: float = 0.0, std: float = 1.0) -> ndarray:
    return np.random.normal(
        loc = mean,
        scale = std,
        size = shape
    )
    

def uniform_init(*shape, low: float = -1.0, high: float = 1.0) -> ndarray:
    return np.random.uniform(
        low = low,
        high = high,
        size = shape
    )


def he_init(*shape) -> np.ndarray:
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


def xavier_init(*shape) -> np.ndarray:
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


def lecun_init(*shape) -> np.ndarray:
    fan_in = shape[0]
    std = np.sqrt(1.0 / fan_in)
    return np.random.randn(*shape) * std


################
#              #
#  ACTIVATION  #
#              #
################

# Sigmoid
def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

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
def softmax(x: ndarray, axis: int = -1) -> ndarray:
    e_x = np.exp(x - np.max(x, axis = axis, keepdims = True))
    return e_x / np.sum(e_x, axis = axis, keepdims = True)

def softmax_derivative(x: ndarray, axis: int = -1) -> ndarray:
    s = softmax(x, axis)
    return s * (1- s)


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


##########
#        #
#  LOSS  #
#        #
##########

# MSE - Mean Squared Error
def mse(y_true: ndarray, y_pred: ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true: ndarray, y_pred: ndarray) -> ndarray:
    return 2 * (y_pred - y_true) / y_true.size


# MAE - Mean Absolute Error
def mae(y_true: ndarray, y_pred: ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def mae_derivative(y_true: ndarray, y_pred: ndarray) -> ndarray:
    return np.sign(y_pred - y_true) / y_true.size


# RMSE - Root Mean Squared error
def rmse(y_true: ndarray, y_pred: ndarray) -> float:
    return np.sqrt(mse(y_true, y_pred))

def rmse_derivative(y_true: ndarray, y_pred: ndarray, eps: float = EPSILON) -> ndarray:
    return (y_pred - y_true) / (rmse(y_true, y_pred) * y_true.size + eps)


# MAPE - Mean Absolute Percentage Error
def mape(y_true: ndarray, y_pred: ndarray, eps: float = EPSILON) -> float:
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def mape_derivative(y_true: ndarray, y_pred: ndarray, eps: float = EPSILON) -> ndarray:
    return (np.sign(y_pred - y_true) / (y_true + eps)) * (100 / y_true.size)


# MSLE - Mean Squared Lagarithmic Error
def msle(y_true: ndarray, y_pred: ndarray) -> float:
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

def msle_derivative(y_true: ndarray, y_pred: ndarray, eps: float = EPSILON) -> ndarray:
    return (2 / y_true.size) * (np.log1p(y_pred + eps) - np.log1p(y_true + eps)) / (y_pred + 1 + eps)


# BCE - Binary Crossentropy
def bce(y_true: ndarray, y_pred: ndarray, from_logits: bool = False, eps: float = EPSILON) -> float:
    if from_logits:
        y_pred = sigmoid(y_pred)

    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_derivative(y_true: ndarray, y_pred: ndarray, from_logits: bool = False, eps: float = EPSILON) -> ndarray:
    if from_logits:
        y_pred = sigmoid(y_pred)
        
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    return y_pred - y_true


# CCE - Categorical Crossentropy
def cce(y_true: ndarray, y_pred: ndarray, from_logits: bool = False, axis: int = -1, eps: float = EPSILON) -> float:
    if from_logits:
        y_pred = softmax(y_pred, axis)

    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cce_derivative(y_true: ndarray, y_pred: ndarray, from_logits: bool = False, axis: int = -1, eps: float = EPSILON) -> ndarray:
    if from_logits:
        y_pred = softmax(y_pred, axis)
        
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    return y_pred - y_true


# SCCE - Sparse Categorical Crossentropy
def scce(y_true: ndarray, y_pred: ndarray, from_logits: bool = False, axis: int = -1, eps: float = EPSILON) -> float:
    if from_logits:
        y_pred = softmax(y_pred, axis)

    y_pred = np.clip(y_pred, eps, 1 - eps)
    batch_size = y_pred.shape[0]
    
    return -np.mean(np.log(y_pred[np.arange(batch_size), y_true]))

def scce_derivative(y_true: ndarray, y_pred: ndarray, from_logits: bool = False, axis: int = -1, eps: float = EPSILON) -> ndarray:
    if from_logits:
        y_pred = softmax(y_pred, axis)

    y_pred = np.clip(y_pred, eps, 1 - eps)
    batch_size = y_pred.shape[0]
    grad = y_pred.copy()
    grad[np.arange(batch_size), y_true] -= 1
    
    return grad