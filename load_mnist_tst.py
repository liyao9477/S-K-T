#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:05:31 2024

@author: boyly
"""

import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784',version=1,as_frame= False)
print(mnist.keys())
# mnist = mnist[mnist['data', 'target','feature_names', 'DESCR', 'details', 'categories','url']]

X,y = mnist["data"],mnist["target"]
print(X.shape)
print(y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")
plt.show()

y[0]
y = y.astype(np.uint8)

X_train,X_test,y_train,y_test = X[:60000],X[60000:],y[:60000],y[60000:]

y_train_5 = (y_train == 5)  # 标记1或0，T or F
y_test_5 = (y_test == 5)

# 使用随机梯度递减分类器
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)

sgd_clf.predict([some_digit])



## Performance Measures -- 有时准确率accuracy无法评估模型优劣

# 1. cross-validation (K-fold cross-validation)
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


from sklearn.base import BaseEstimator  # 估计器

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X),1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy" )

# 2. Confusion Matrix
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) # result of: T or F

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

# precision and recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) # 预测为正例中，实际是正的占比。预测为敏感的客户里实际有多少真正敏感；
recall_score(y_train_5, y_train_pred)   # 所有实际为正的案例，预测准确的占比。所有敏感客户被预测为敏感的占比。

# F1-score # harmonic mean of two metrics above
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

# Precision/Recall Trade-off -- 根据得分分值判断类别，注意阈值不能调整
y_score = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')

from sklearn.metrics import precision_recall_curve
precision, recalls, thresholds = precision_recall_curve(y_train_5, y_score)

def plot_precision_recall_vs_threshold(precision,recalls,thresholds):
    plt.plot(thresholds,precision[:-1],"b--", label = "precision")
    plt.plot(thresholds,recalls[:-1],"g-", label = "recall")

plot_precision_recall_vs_threshold(precision, recalls, thresholds)
plt.show()

# 如何控制精准率大于90%
threshold_90_precision = thresholds[np.argmax(precision >=0.90)]
y_train_pred_90 = (y_score >= threshold_90_precision)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_score)

def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth=2, label = label)
    plt.plot([0,1],[0,1],'k--')
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_score)

## 对比随机森林的ROC曲线
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')
y_score_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_score_forest)

plt.plot(fpr, tpr,"b:",label = "SGD")
plot_roc_curve(fpr_forest, tpr_forest,"Random Forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_5, y_score_forest)


### multiclass Classification 多分类，OvR or OvO
from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.predict([some_digit])

some_digit_score = svm_clf.decision_function([some_digit])
some_digit_score   # OvO strategy, 10 scores
np.argmax(some_digit_score)
svm_clf.classes_
svm_clf.classes_[5]

from sklearn.multiclass import OneVsRestClassifier  # OvR strategy
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
sgd_clf.decision_function([some_digit])
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

#### error analysis
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis = 1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap = plt.cm.gray)
plt.show()





