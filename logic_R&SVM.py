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


# SVM
# Linear SVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

X = iris["data"][:,(2,3)]  
y = (iris["target"] == 2).astype(np.int64)

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
    ])
svm_clf.fit(X, y)

svm_clf.predict([[4.71,1.63]])

svm_clf.predict([[5.71,1.63]])

# Nonlinear SVM
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15)
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ])
polynomial_svm_clf.fit(X, y)

polynomial_svm_clf.predict([[4.71,1.63]])

polynomial_svm_clf.predict([[5.71,1.63]])

# SVM regresiong --> SVR
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)








