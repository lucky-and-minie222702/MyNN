from ..Core import *
from .Operations import *
from . import Functional as F
from . import Layers
from . import Optimizers
from . import Losses
from . import Modules
from tqdm import tqdm

class Trainer:
    def __init__(self, max_ncols: int = 110):
        self.MAX_NCOLS = max_ncols

class SequentialTrainer(Trainer):
    def __init__(self, model: Modules.Module | Modules.SequentialModule | Modules.DynamicModule, optimizer: Optimizers.Optimizer, loss: Losses.Loss, **kwargs):
        super().__init__(**kwargs)
        
        self.model = model
        
        self.optimizer = optimizer
        if isinstance(self.optimizer, str):
            self.optimizer = Optimizers.optimizer_byname(self.optimizer)
        self.optimizer.assign(self.model)
        
        self.loss = loss
        if isinstance(self.loss, str):
            self.loss = Losses.loss_byname(self.loss)
        
    def train_on_batch(self, X_batch: ndarray, y_batch: ndarray) -> Tuple[float, ndarray]:
        prediction = self.model.forward(X_batch)
        
        loss = self.loss.forward(y_batch, prediction)
        
        self.model.backward(self.loss.backward())
        
        self.optimizer.step()
        
        return loss, prediction
    
    def generate_batches(self, length: int, batch_size: int, shuffle: bool = True):
        while True:
            if shuffle:
                indices = np.random.permutation(length)
            else:
                indices = np.arange(length)
            
            for i in range(0, length, batch_size):
                batch_indices = indices[i:i+batch_size:]
                yield batch_indices
                
    def get_prediciton(self, inp: ndarray, batch_size: int = 32, verbose: bool = True) -> ndarray:
        generator = self.generate_batches(len(inp), batch_size, shuffle = False)
        
        steps = int(np.ceil(len(inp) / batch_size))
        step_length = len(str(steps))
        
        total_predictions = []
        
        pbar = range(steps)
        if verbose:
            pbar = tqdm(range(steps), bar_format = "{desc} ETA: {remaining} |{bar}|", ncols = self.MAX_NCOLS)
        
        for step in pbar:
            batch_indices = next(generator)
            total_predictions.append(self.model.forward(inp[batch_indices], training = False))
            
            if verbose:
                pbar.set_description(f"{step:>{step_length}}/{steps:>{step_length}} |> ")
        
        return np.concatenate(total_predictions, axis = 0)
        
    def fit(self, 
            X: ndarray, y: ndarray,
            epochs: int,
            batch_size: int,
            metrics: Dict[str, Callable[[ndarray, ndarray], ndarray]] = {},
            val_data: Tuple[ndarray, ndarray] | None = None,
            val_batch_size: int | None = None,
            shuffle: bool = True,
        ):
        
        assert len(X) == len(y), f"Inequal length: X {X.shape} - y {y.shape}"
        
        step_per_epoch = int(np.ceil(len(X) / batch_size))
        step_length = len(str(step_per_epoch))
        
        val_batch_size = batch_size if val_batch_size is None else val_batch_size

        generator = self.generate_batches(len(X), batch_size, shuffle = shuffle)
        
        history = {f"train_{k}": [] for k in metrics.keys()}
        history["train_loss"] = []
        if val_data is not None:
            history.update({f"val_{k}": [] for k in metrics.keys()})
            history["val_loss"] = []
        
        
        # train
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}:")
            pbar = tqdm(range(step_per_epoch), bar_format = " Train {desc} ETA: {remaining} |{bar}|", ncols =  self.MAX_NCOLS)

            total_loss = 0
            total_metrics = {k: 0 for k in metrics.keys()}

            for step in pbar:
                batch_indices = next(generator)
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                
                loss, prediction = self.train_on_batch(X_batch, y_batch)

                total_loss += loss
                
                for name in total_metrics.keys():
                    total_metrics[name] += metrics[name](y_batch, prediction)
                list_metrics = [f"{name}: {value / (step + 1):.4f}" for name, value in total_metrics.items()]
                
                postfix = f"loss: {total_loss / (step + 1):.4f} " + " | ".join(list_metrics)
                pbar.set_description(f"{step+1:>{step_length}}/{step_per_epoch:>{step_length}} |> " + postfix)

            history["train_loss"].append(total_loss / step_per_epoch)
            for k in metrics.keys():
                history["train_" + k].append(total_metrics[k] / step_per_epoch)

                
            # validation
            if val_data is not None:
                prediciton = self.get_prediciton(val_data[0], val_batch_size, verbose = False)
                
                val_loss = self.loss.compute_output(val_data[1], prediciton)
                
                val_metrics = {name: f(val_data[1], prediciton) for name, f in metrics.items()}
                list_val_metrics = [f"{name}: {value:.4f}" for name, value in val_metrics.items()]
                
                print(f" Val     {' ' * step_length * 2}|>", f"loss: {val_loss:.4f} " + " | ".join(list_val_metrics))
                
                history["val_loss"].append(val_loss)
                for k in metrics.keys():
                    history["val_" + k].append(val_metrics[k])
                
        return history