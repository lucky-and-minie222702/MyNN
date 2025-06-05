import numpy as np
# import Minnie
# from Minnie.NN import Layer

# model = Layer.Sequential([
#     Layer.Dense(10, 20, "relu"),
#     Layer.Dense(20, 30, "relu", bias = False),
#     Layer.Dense(30, 15, "relu")
# ])

# # dummy
# X = np.random.rand(100, 10)
# y = model.forward(X)

# print(y)

class A:
    def __init__(self, w):
        self.w = w
        w -= 10
        
class B:
    def __init__(self, w):
        self.w = w
        
    def get_w(self):
        return self.w
        
model = B(np.ones(10))
print(model.w)
A(model.get_w())
print(model.w)