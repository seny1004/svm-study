import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


import os

data = pd.read_csv("seattle-weather.csv")

col_names = data.columns

from sklearn.model_selection import train_test_split

X = data.drop(["weather","date"], axis= 1)
y = data["weather"]
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

cols = X_train.columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# class SVM:
#
#     def __init__(self, learning_rate = 0.001,lambda_param=0.01, n_iters = 1000):
#         self.lr = learning_rate
#         self.lambda_param = lambda_param
#         self.n_iters = n_iters
#         self.w = None
#         self.b = None
#
#     def fit(self,X, y):
#         n_samples, n_features = X.shape
#
#         y_ = np.where(y <= 0, -1, 1)
#
#         self.w = np.zeros(n_features)
#         self.b = 0
#
#         for _ in range(self.n_iters):
#             for idx, x_i in enumerate(X):
#                 condition = y_[idx] * (np.dot(x_i, self.w)-self.b) >= 1
#                 if condition:
#                     self.w -= self.lr*(2*self.lambda_param * self.w)
#                 else:
#                     self.w -= self.lr*(2*self.lambda_param * self.w - np.dot(x_i,y_[idx]))
#                     self.b -= self.lr*y_[idx]
#     def predict(self, X):
#         approx = np.dot(X, self.w) - self.b
#         return np.sign(approx)
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(accuracy_score(y_test, y_pred))
