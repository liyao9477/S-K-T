#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 08:13:25 2024

@author: boyly
"""

import numpy as np
## 根据正规方程可以直接得到权重β的最优解 β = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100,1)
y = 4 + X * 3 + np.random.rand(100,1)*0.01


lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.intercept_, lin_reg.coef_

X_new = np.array([[1],[2]])
lin_reg.predict(X_new)


## learning schedule 随机梯度下降SGD，学习率的计划
n=50
t0,t1 = 5,50 # 超参数
β = np.random.randn(2,1)
m = 100 # 迭代次数

def learning_schedule(t):
    return t0 / (t+t1)

for e in range(n):
    for i in range(m):
        r_index = np.random.randin(i)
        Xi = X[r_index:r_index+1]
        yi = y[r_index:r_index+1]
        gradients = 2 * Xi.T.dot(Xi.dot(β) - yi)
        eta = learning_schedule(e * m+i)
        β = β - eta * gradients


from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1) # run 1000 epochs until loss drop less than 0.001
# learning rate = 0.1
sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_

## 总结：简单线性回归算法：Normal Equatioin , SVD, Batch GD, SGD, Mini-batch GD

# 多项式回归
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
X_poly[0]  # X_poly 包含了X，和X平方两个特征

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_   # y = 0.48*X^2 + 1.02*X +2.06 + noise  与原方程的系数接近

## 学习曲线
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [],[]
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors),'r-+',linewidth = 2,label ='train')
    plt.plot(np.sqrt(val_errors),'b-',linewidth = 3,label ='val')

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)  # 通过增加训练数据，解决underfitting

# 调整模型复杂度
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline([
    ("poly_features",PolynomialFeatures(degree=10,include_bias=False)),
    ("lin_reg", LinearRegression())
    ])

plot_learning_curves(polynomial_regression, X, y)


# Ridge regressioin 岭回归，在损失函数中加入一个正则化项（l2），降低模型复杂度，防止过拟合
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])  # 设置penalty超参数，使SGD模型增加l2正则化项，实现简单的岭回归
# penalty不是sklearn.linear_model.LinearRegression类的标准功能。
# penalty参数用于指定正则化项的类型，L1正则化倾向于产生稀疏的权重（即许多权重为零），这有助于特征选择；
# 而L2正则化则倾向于使权重值均匀分布，避免过大的权重值出现，从而防止过拟合。

# Lasso regression 套索回归，在损失函数中增加一个l1正则化项，不仅能够降低模型的复杂度，还能实现特征选择。
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]]) 

# Elastic Net 弹性网络, 是岭回归和套索回归的结合
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]]) 










