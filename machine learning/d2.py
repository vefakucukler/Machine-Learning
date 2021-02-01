# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 02:18:16 2019

@author: Dr Hacı Abi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

plt.style.use('ggplot')

iris = datasets.load_iris()

X = iris.data[:, [0,2]]
y = iris.target

plt.figure()
plt.scatter(X[y==0,0],X[y==0,1],c="b")
plt.scatter(X[y==1,0],X[y==1,1],c='r')
plt.scatter(X[y==2,0],X[y==2,1],c="g")

plt.legend(iris["target_names"])
plt.xlabel(iris.feature_names[0]);
plt.ylabel(iris.feature_names[2]);

def plot_decision_region(X, y, classifier, legend=[],resolution=0.02):


    # marker ve renk seçimi
    # burada sınıf sayısı kadar modifiye etmek gerekiyor aksi takdirde hata alınır.
    markers = ('o', 'o', 'o')
    colors = ('red', 'blue', 'green')

    # karar bölgesi ayarlanıyor
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap="coolwarm")
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # legend kısmını düzgün göstermek için eklendi
    line_list = []
    # örnekler çizdiriliyor
    for idx, cl in enumerate(np.unique(y)):
        dummy = plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl)
        line_list.append(dummy)

    plt.legend(line_list,legend)

kNN = KNeighborsClassifier(n_neighbors=3,metric="euclidean")
kNN.fit(X,y)

plt.figure()
plot_decision_region(X,y,kNN,legend=iris.target_names)

plt.xlabel(iris.feature_names[0]);
plt.ylabel(iris.feature_names[2]);
plt.title("kNN'de en yakın 3 komşu kullanıldı");

kNN = KNeighborsClassifier(n_neighbors=3,metric="euclidean")
kNN.fit(X,y)

plt.figure()
plot_decision_region(X,y,kNN,legend=iris.target_names)

plt.xlabel(iris.feature_names[0]);
plt.ylabel(iris.feature_names[2]);
plt.title("kNN'de en yakın 3 komşu kullanıldı");

kNN = KNeighborsClassifier(n_neighbors=21,metric="chebyshev")
kNN.fit(X,y)

plt.figure()
plot_decision_region(X,y,kNN,legend=iris.target_names)

plt.xlabel(iris.feature_names[0]);
plt.ylabel(iris.feature_names[2]);
plt.title("kNN'de en yakın 3 komşu kullanıldı");