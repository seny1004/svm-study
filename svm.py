import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

class SVM:

    def __init__(self, learning_rate = 0.001,lambda_param=0.01, n_iters = 1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self,X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        #gradien decent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                print(idx, x_i)

                #true or false
                condition = y_[idx] * (np.dot(x_i, self.w)-self.b) >= 1

                if condition:
                    self.w -= self.lr*(2 * self.lambda_param * self.w)

                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i,y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
