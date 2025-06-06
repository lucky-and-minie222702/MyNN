import numpy as np
import pandas as pd
from NHCT import Core, Math
from NHCT.NN import *
from NHCT.NN.Utils import *
import sklearn.metrics as met
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# code for: https://www.kaggle.com/competitions/digit-recognizer/
# result 0.97003 on kaggle
# all the dataset files are saved in test_data folder

np.random.seed(42)

model = Modules.SequentialModule([
    Layers.Dense(784, 128, "relu"),
    Layers.Dropout(rate = 0.3),
    Layers.Dense(128, 128, "relu"),
    Layers.Dropout(rate = 0.5),
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
    optimizer = "sgd",
    loss = "scce"
)

hist = trainer.fit(
    X_train, y_train,
    epochs = 35,
    batch_size = 32,
    metrics = {
        "accuracy": lambda y_true, y_pred: met.accuracy_score(y_true, np.argmax(y_pred, axis = -1)),
    },
    val_data = (X_val, y_val)
)

trainer.model.save_pickle("test_weights.pkl")
trainer.model.load_pickle("test_weights.pkl")

prediction = trainer.get_prediciton(X_val)
# check the model saving
print("Val accuracy:", met.accuracy_score(y_val, np.argmax(prediction, axis = -1)))

X_test = pd.read_csv("test_data/test.csv")

X_test = X_test
X_test = X_test.to_numpy() / 255.0

test_prediction = trainer.get_prediciton(X_test)
ans_df = pd.DataFrame({
    "ImageId": np.arange(1, len(test_prediction) + 1),
    "Label": np.argmax(test_prediction, axis = -1)
})

ans_df.to_csv("test_submission.csv", index = False)

plt.plot(hist["val_loss"], label = "val_loss")
plt.plot(hist["train_loss"], label = "train_loss")
plt.grid()
plt.legend()
plt.savefig("test_model_loss.png")
plt.close()

plt.plot(hist["val_accuracy"], label = "val_accuracy")
plt.plot(hist["train_accuracy"], label = "train_accuracy")
plt.grid()
plt.legend()
plt.savefig("test_model_accuracy.png")
plt.close()