from ..Core import *
from scipy.special import erf

#################
#               #
#  Initializer  #
#               #
#################


def normal_init(*shape, mean: float = 0.0, std: float = 1.0) -> JArray:
    return jnp.asarray(np.random.normal(
        loc = mean,
        scale = std,
        size = shape
    ))
    

def uniform_init(*shape, low: float = -1.0, high: float = 1.0) -> JArray:
    return jnp.asarray(np.random.uniform(
        low = low,
        high = high,
        size = shape
    ))


def he_init(*shape) -> JArray:
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return jnp.asarray(np.random.randn(*shape) * std)


def xavier_init(*shape) -> JArray:
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return jnp.asarray(np.random.uniform(-limit, limit, size=shape))


def lecun_init(*shape) -> JArray:
    fan_in = shape[0]
    std = np.sqrt(1.0 / fan_in)
    return jnp.asarray(np.random.randn(*shape) * std)


################
#              #
#  ACTIVATION  #
#              #
################

# Sigmoid
def sigmoid(x: JArray) -> JArray:
    return 1 / (1 + jnp.exp(-jnp.clip(x, -100, 100)))

def sigmoid_derivative(x: JArray) -> JArray:
    s = sigmoid(x)
    return s * (1 - s)

# Tanh
def tanh(x: JArray) -> JArray:
    return jnp.tanh(x)

def tanh_derivative(x: JArray) -> JArray:
    return 1 - jnp.tanh(x) ** 2


# ReLU
def relu(x: JArray) -> JArray:
    return jnp.maximum(0, x)

def relu_derivative(x: JArray) -> JArray:
    return (x > 0).astype(float)


# Leaky ReLU
def leaky_relu(x: JArray, alpha: float) -> JArray:
    return jnp.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: JArray, alpha: float) -> JArray:
    return jnp.where(x > 0, 1.0, alpha)


# GeLU
def gelu(x: JArray) -> JArray:
    return 0.5 * x * (1 + erf(x / jnp.sqrt(2)))

def gelu_derivative(x: JArray) -> JArray:
    phi = jnp.exp(-0.5 * x**2) / jnp.sqrt(2 * jnp.pi)
    return 0.5 * (1 + erf(x / jnp.sqrt(2))) + 0.5 * x * phi


# SiLU (Swish)
def silu(x: JArray) -> JArray:
    return x * sigmoid(x)

def silu_derivative(x: JArray) -> JArray:
    s = sigmoid(x)
    return s * (1 + x * (1 - s))


# Softplus
def softplus(x: JArray) -> JArray:
    return jnp.log1p(jnp.exp(x))

def softplus_derivative(x: JArray) -> JArray:
    return sigmoid(x)


# Softsign
def softsign(x: JArray) -> JArray:
    return x / (1 + jnp.abs(x))

def softsign_derivative(x: JArray) -> JArray:
    return 1 / (1 + jnp.abs(x))**2


# Softmax
def softmax(x: JArray, axis: int = -1) -> JArray:
    e_x = jnp.exp(x - jnp.max(x, axis = axis, keepdims = True))
    return e_x / jnp.sum(e_x, axis = axis, keepdims = True)

def softmax_derivative(x: JArray, axis: int = -1) -> JArray:
    s = softmax(x, axis)
    return s * (1- s)


# SeLU
SELU_ALPHA = 1.6732632423543772
SELU_LAMBDA = 1.0507009873554805

def selu(x: JArray) -> JArray:
    return SELU_LAMBDA * jnp.where(x > 0, x, SELU_ALPHA * (jnp.exp(x) - 1))

def selu_derivative(x: JArray) -> JArray:
    return SELU_LAMBDA * jnp.where(x > 0, 1.0, SELU_ALPHA * jnp.exp(x))


# CeLU
def celu(x: JArray, alpha: float = 1.0) -> JArray:
    return jnp.where(x >= 0, x, alpha * (jnp.exp(x / alpha) - 1))

def celu_derivative(x: JArray, alpha: float = 1.0) -> JArray:
    return jnp.where(x >= 0, 1.0, jnp.exp(x / alpha))


##########
#        #
#  LOSS  #
#        #
##########

# MSE - Mean Squared Error
def mse_loss(y_true: JArray, y_pred: JArray) -> float:
    return jnp.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true: JArray, y_pred: JArray) -> JArray:
    return 2 * (y_pred - y_true) / y_true.size


# MAE - Mean Absolute Error
def mae_loss(y_true: JArray, y_pred: JArray) -> float:
    return jnp.mean(jnp.abs(y_true - y_pred))

def mae_derivative(y_true: JArray, y_pred: JArray) -> JArray:
    return jnp.sign(y_pred - y_true) / y_true.size


# RMSE - Root Mean Squared error
def rmse_loss(y_true: JArray, y_pred: JArray) -> float:
    return jnp.sqrt(mse_loss(y_true, y_pred))

def rmse_derivative(y_true: JArray, y_pred: JArray, eps: float = EPSILON) -> JArray:
    return (y_pred - y_true) / (rmse_loss(y_true, y_pred) * y_true.size + eps)


# MAPE - Mean Absolute Percentage Error
def mape_loss(y_true: JArray, y_pred: JArray, eps: float = EPSILON) -> float:
    return jnp.mean(jnp.abs((y_true - y_pred) / (y_true + eps))) * 100

def mape_derivative(y_true: JArray, y_pred: JArray, eps: float = EPSILON) -> JArray:
    return (jnp.sign(y_pred - y_true) / (y_true + eps)) * (100 / y_true.size)


# MSLE - Mean Squared Lagarithmic Error
def msle_loss(y_true: JArray, y_pred: JArray) -> float:
    return jnp.mean((jnp.log1p(y_true) - jnp.log1p(y_pred)) ** 2)

def msle_derivative(y_true: JArray, y_pred: JArray, eps: float = EPSILON) -> JArray:
    return (2 / y_true.size) * (jnp.log1p(y_pred + eps) - jnp.log1p(y_true + eps)) / (y_pred + 1 + eps)


# BCE - Binary Crossentropy
def bce_loss(y_true: JArray, y_pred: JArray, from_logits: bool = False, eps: float = EPSILON) -> float:
    if from_logits:
        y_pred = sigmoid(y_pred)

    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def bce_derivative(y_true: JArray, y_pred: JArray, from_logits: bool = False, eps: float = EPSILON) -> JArray:
    if from_logits:
        y_pred = sigmoid(y_pred)
        
    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    
    return y_pred - y_true


# CCE - Categorical Crossentropy
def cce_loss(y_true: JArray, y_pred: JArray, from_logits: bool = False, axis: int = -1, eps: float = EPSILON) -> float:
    if from_logits:
        y_pred = softmax(y_pred, axis)

    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=1))

def cce_derivative(y_true: JArray, y_pred: JArray, from_logits: bool = False, axis: int = -1, eps: float = EPSILON) -> JArray:
    if from_logits:
        y_pred = softmax(y_pred, axis)
        
    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    
    return y_pred - y_true


# SCCE - Sparse Categorical Crossentropy
def scce_loss(y_true: JArray, y_pred: JArray, from_logits: bool = False, axis: int = -1, eps: float = EPSILON) -> float:
    if from_logits:
        y_pred = softmax(y_pred, axis)

    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    batch_size = y_pred.shape[0]
    
    return -jnp.mean(jnp.log(y_pred[jnp.arange(batch_size), y_true]))

def scce_derivative(y_true: JArray, y_pred: JArray, from_logits: bool = False, axis: int = -1, eps: float = EPSILON) -> JArray:
    if from_logits:
        y_pred = softmax(y_pred, axis)

    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    batch_size = y_pred.shape[0]
    grad = y_pred.at[jnp.arange(batch_size), y_true].add(-1.0)
    
    return grad