import pandas as pd
from NHCT import Core, Math
from NHCT.NN import *
from NHCT.NN.Utils import *
from NHCT import Math
import sklearn.metrics as met
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

print("Jax default:", jax.default_backend(), "devices:", jax.devices())

model = Modules.SequentialModule([
    Layers.Dense(128, 32, "relu", dropout = 0.5),
    Layers.Dense(32, 10, "softmax"),
])
model.build()
print(model.get_layer_opt_names())

X, y = make_classification(n_samples = 200_000,
                           n_features = 128,
                           n_informative = 100, 
                           n_classes = 10, 
                           random_state = 42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

trainer = SequentialTrainer(
    model = model,
    optimizer = Optimizers.optimizer_byname("sgd", model, momentum = 0.0),
    loss = "scce"
)

hist = trainer.fit(
    X_train, y_train,
    epochs = 100,
    batch_size = 64,
    metrics = {
        "accuracy": lambda y_true, y_pred: met.accuracy_score(y_true, np.argmax(y_pred, axis = -1)),
        "topk=3": lambda y_true, y_pred: met.top_k_accuracy_score(y_true, y_pred, k = 3, labels = np.arange(10)),
    },
    val_data = (X_val, y_val),
)

trainer.model.save_pickle("test_weights.pkl")
trainer.model.load_pickle("test_weights.pkl")

prediction = trainer.get_prediciton(X_val)
# check the model saving
print("Val accuracy:", met.accuracy_score(y_val, np.argmax(prediction, axis = -1)))