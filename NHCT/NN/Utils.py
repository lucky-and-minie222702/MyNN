from ..Core import *
from .Operations import *
from . import Functional as F
from . import Layers
from . import Optimizers
from . import Losses
from . import Modules


def format_time_units(seconds):
    ms = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs:
        parts.append(f"{secs}s")
    if ms:
        parts.append(f"{ms}ms")
    
    return ' '.join(parts) or "0 ms"


class BatchIndicesGenerator:
    def __init__(self, length: int, batch_size: int, shuffle: bool = True):
        self.length = length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_steps = int(np.ceil(self.length / batch_size))

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.length)
        else:
            indices = np.arange(self.length)

        for i in range(0, self.length, self.batch_size):
            yield indices[i:i + self.batch_size]
            
class BatchGenerator:
    def __init__(self, *args, batch_size: int, shuffle: bool = True, to_jax: bool = False):
        self.data = []
        for d in args:
            self.data.append(d)
            
        self.to_jax = to_jax
            
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.length = len(self.data[0])
        self.num_steps = int(np.ceil(self.length / batch_size))

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.length)
        else:
            indices = np.arange(self.length)

        for i in range(0, self.length, self.batch_size):
            if self.to_jax:
                yield (jnp.asarray(d[indices[i:i + self.batch_size]]) for d in self.data)
            else:
                yield (d[indices[i:i + self.batch_size]] for d in self.data)


class Trainer:
    def __init__(self):
        pass

class SequentialTrainer(Trainer):
    def __init__(self, model: Modules.Module, optimizer: Optimizers.Optimizer, loss: Losses.Loss, **kwargs):
        super().__init__(**kwargs)
        
        self.model = model
        
        self.optimizer = optimizer
        if isinstance(self.optimizer, str):
            self.optimizer = Optimizers.optimizer_byname(self.optimizer, module = model)
        
        self.loss = loss
        if isinstance(self.loss, str):
            self.loss = Losses.loss_byname(self.loss)

        
    def train_on_batch(self, X_batch: ndarray, y_batch: ndarray) -> Tuple[float, ndarray]:
        prediction = self.model.forward(X_batch)
        loss = self.loss.forward(y_batch, prediction)
        
        self.model.backward(self.loss.backward())
        
        self.optimizer.step()
        
        # loss and prediction on batch
        return loss, prediction
    
    def generate_batches(self, X: ndarray, y: ndarray, batch_size: int, shuffle: bool = True):
        return BatchGenerator(X, y, batch_size = batch_size, shuffle = shuffle, to_jax = True)
                
    def get_prediciton(self, inp: ndarray, batch_size: int = 32, verbose: bool = True) -> ndarray:
        inp = jnp.asarray(inp)
        generator = BatchIndicesGenerator(len(inp), batch_size, shuffle = False)
        
        total_predictions = []
        
        for step, batch_indices in enumerate(generator):
            total_predictions.append(self.model.forward(inp[batch_indices], training = False))
        
        return np.concatenate(total_predictions, axis = 0)