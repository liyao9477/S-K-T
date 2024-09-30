#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:01:47 2024

@author: boyly
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:,3:]  # 筛选一个特征进行判别
y = (iris["target"] == 2).astype(np.int8)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

def plot_tpm():
    plt.plot(X_new, y_proba[:,1], "g-", label="Iris virginica")
    plt.plot(X_new, y_proba[:,0], "b--", label="Not Iris virginica")
plot_tpm()    
plt.show()

log_reg.predict([[1.71],[1.43]])



























