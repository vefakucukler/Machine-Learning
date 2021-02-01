# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 02:54:13 2019

@author: Dr Hacı Abi
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_svm_decision_region_2d(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=2, facecolors='none',edgecolor="k",alpha=0.5);
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

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

def generate_noisy_line(a=0.5, b=5, y_noise=5, random_seed=[], x=np.linspace(-10, 50, 100)):
    if(random_seed):
        # Hep aynı değerleri oluşturması için seed değerini sabitleyelim
        np.random.seed(42)
    else:
        pass

    # y noktaları lineer bir doğrunun üzerine rastgele gürültü eklenmesi ile
    y = (a * x + b) + np.random.rand(len(x)) * y_noise

    return x,y