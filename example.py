# run on this kaggle competition: https://www.kaggle.com/competitions/digit-recognizer/
# public score: 0.97207


import pandas as pd

from NHCT import Core, Math
from NHCT.NN import *
from NHCT.NN.Utils import *

import sklearn.metrics as met
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tqdm import tqdm

np.random.seed(42)

print("Devices:", jax.devices())

model = Modules.SequentialModule([
    Layers.Dense(784, 128, "relu", dropout = 0.3),
    Layers.Dense(128, 128, "relu", dropout = 0.5),
    Layers.Dense(128, 10, "softmax"),
])
model.build()


X = pd.read_csv("test_data/train.csv")
y = X["label"].to_numpy()
X = X.drop(columns = "label")
X = X.to_numpy() / 255.0

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

trainer = SequentialTrainer(
    model = model,
    optimizer = Optimizers.optimizer_byname("sgd", model, momentum = 0.2),
    loss = "scce"
)

epochs = 40
for ep in range(1, epochs + 1):
    print(f"Epoch: {ep}/{epochs}")
    for X_batch, y_batch in tqdm(trainer.generate_batches(X_train, y_train, batch_size = 32), ncols = 60):
        loss, _ = trainer.train_on_batch(X_batch, y_batch)
    
    pred = trainer.get_prediciton(X_val, batch_size = 128)
    val_loss = trainer.loss.compute_output(y_val, pred)
    
    print(f" Val: loss={val_loss:.4f}, accuracy={met.accuracy_score(y_val, np.argmax(pred, axis = -1)):.4f}")

model.save_pickle("test_weights.pkl")
model.load_pickle("test_weights.pkl")

X_test = pd.read_csv("test_data/test.csv")

X_test = X_test
X_test = X_test.to_numpy() / 255.0

test_prediction = trainer.get_prediciton(X_test)
ans_df = pd.DataFrame({
    "ImageId": np.arange(1, len(test_prediction) + 1),
    "Label": np.argmax(test_prediction, axis = -1)
})

ans_df.to_csv("test_submission.csv", index = False)

prediction = trainer.get_prediciton(X_val)
# check the model saving
print("Val accuracy:", met.accuracy_score(y_val, np.argmax(prediction, axis = -1)))