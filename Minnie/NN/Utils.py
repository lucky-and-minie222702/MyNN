from ..LibImport import *
from .Operation import *
from . import Functional as F
from .Layer import Layer
from .Optimizer import Optimizer
from .Loss import Loss
from .Module import Module
from tqdm import tqdm


class Trainer:
    def __init__(self, model: Module, optimizer: Optimizer, loss: Loss):
        self.model = model
        
        self.optimizer = optimizer
        self.optimizer.assign(self.model)
        
        self.loss = loss
        
        self.indices = None
        
    def train_on_batch(self, X_batch: ndarray, y_batch: ndarray):
        prediction = self.model.forward(X_batch)
        
        loss = self.loss.forward(y_batch, prediction)
        
        self.model.backward(self.loss.backward())
        
        self.optimizer.step()
        
        return loss, prediction
    
    def generate_batches(self, X: ndarray, y: ndarray, batch_size: int):
        while True:
            self.indices = np.random.permutation(len(X))
            
            for i in range(0, len(X), batch_size):
                batch_indices = self.indices[i:i+batch_size:]
                yield X[batch_indices], y[batch_indices]
            
    def fit(self, 
            X: ndarray, y: ndarray,
            epochs: int,
            batch_size: int,
            metrics: Dict[str, Callable[[ndarray, ndarray], ndarray]],
            step_per_epoch: int | None = None
        ):
        
        if step_per_epoch is None:
            step_per_epoch = int(np.ceil(len(X) / batch_size))
        
        generator = self.generate_batches(X, y, batch_size)
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}:")
            pbar = tqdm(range(step_per_epoch), bar_format = "{l_bar}{bar} | ETA: {remaining}")

            total_loss = 0
            total_mets = {k: 0 for k in metrics.keys()}

            for step in pbar:
                X_batch, y_batch = next(generator)
                loss, prediction = self.train_on_batch(X_batch, y_batch)
                total_loss += loss
                
                for name in total_mets.keys():
                    total_mets[name] += metrics[name](y_batch, prediction)
                
                mets = [f"{name}: {val / (step + 1):.4f}" for name, val in total_mets.items()]

                postfix = f"loss: {total_loss / (step + 1):.4f} " + " | ".join(mets)
                pbar.set_description(f"step {step}/{step_per_epoch} > " + postfix)
                
            print(self.model.layer_params[-1][0])