# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 01:01:55 2019

@author: Dr Hacı Abi
"""
import matplotlib.pyplot as plt
import seaborn as sns
import random
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=200, # veri setinde kaç örnek olcak
                n_features = 2, # veri setinin 2 boyutlu olmasını istiyoruz
                 centers = 4, # deneme amaçlı k=4 olsun
                 random_state=42, # tekrarlanabilir olması açısından
                )

sns.set_style("darkgrid")

plt.figure()
plt.scatter(X[:,0],X[:,1],c="k",alpha=0.3);
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("4 adet sınıflandırılmamış küme");

k = 4
random.seed(42)
merkezler = random.sample(list(X),k)


cm_bright = ListedColormap(['red','blue','lime','yellow'])
kume_id  = [0,1,2,3]

plt.figure()
plt.scatter(X[:,0],X[:,1],c="k",alpha=0.15);
# Seçilen noktaları çizdiriyoruz
plt.scatter([x[0] for x in merkezler],[x[1] for x in merkezler],
           cmap=cm_bright,
           c=kume_id,alpha=1,s=200,marker=".",edgecolor="k")
# Eksen etiketlemeleri
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("1. iterasyon / 1. adım sonu - İlklendirilmiş merkezler");


mesafeler = cdist(X, merkezler)


kumeler=np.argmin(mesafeler,axis=1)

plt.figure()
plt.scatter(X[:,0],X[:,1],c=kumeler,alpha=0.15,cmap=cm_bright);

plt.scatter([x[0] for x in merkezler],[x[1] for x in merkezler],
           cmap=cm_bright,
           c=kume_id,alpha=1,s=200,marker=".",edgecolor="k")

plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("1. iterasyon / 2. adım sonu - Sınıflara atanmış örnekler");

yeni_merkezler = []
for kume in kume_id:
    yeni_merkezler.append(X[kumeler==kume].mean(0))

plt.figure()
plt.scatter(X[:,0],X[:,1],c=kumeler,alpha=0.15,cmap=cm_bright);
# Seçilen noktaları çizdiriyoruz
plt.scatter([x[0] for x in merkezler],[x[1] for x in merkezler],
           cmap=cm_bright,
           c=kume_id,alpha=0.3,s=200,marker=".",edgecolor="k")

plt.scatter([x[0] for x in yeni_merkezler],[x[1] for x in yeni_merkezler],
           cmap=cm_bright,
           c=kume_id,alpha=1,s=200,marker=".",edgecolor="k")

for kume in kume_id:
    plt.plot([merkezler[kume][0],yeni_merkezler[kume][0]],
            [merkezler[kume][1],yeni_merkezler[kume][1]],c=cm_bright.colors[kume])

# Eksen etiketlemeleri
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("1. iterasyon / 2. adım sonu - Sınıflara atanmış örnekler");

yeni_merkezler = np.array([X[kumeler==kume].mean(0) for kume in kume_id])

def kmeans(X, k=4, max_iterasyon = 300):
    # 1. adım
    merkezler = np.array(random.sample(list(X),k))
    for iter in range(max_iterasyon):
        # 2. adım
        kumeler=np.argmin(cdist(X, merkezler),axis=1)
        # 3. adım
        merkezler = np.array([X[kumeler==kume].mean(0) for kume in kume_id])
    return kumeler,merkezler

kumeler, merkezler = kmeans(X)

plt.figure()
plt.scatter(X[:,0],X[:,1],c=kumeler,alpha=0.15,cmap=cm_bright,edgecolor="k");
# Seçilen noktaları çizdiriyoruz
plt.scatter([x[0] for x in merkezler],[x[1] for x in merkezler],
           cmap=cm_bright,
           c=kume_id,s=200,marker=".",edgecolor="k")


# Eksen etiketlemeleri
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("1. iterasyon / 2. adım sonu - Sınıflara atanmış örnekler");

def kmeans_fun(X, k=4, max_iterasyon = 300):

    random.seed(42)


    merkezler_array = []
    merkezler = np.array(random.sample(list(X),k))
    merkezler_array.append(merkezler)
    for iter in range(max_iterasyon):

        kumeler=np.argmin(cdist(X, merkezler),axis=1)

        merkezler = np.array([X[kumeler==kume].mean(0) for kume in kume_id])
        merkezler_array.append(merkezler)
    return kumeler,merkezler_array

kumeler, merkezler_array = kmeans_fun(X)

plt.figure()
plt.scatter(X[:,0],X[:,1],c=kumeler,alpha=0.15,cmap=cm_bright,edgecolor="k");

# merkezleri çizdiriyoruz
for i in range(len(merkezler_array)):

    if i==0:
        current_marker = "s"
    elif i==(len(merkezler_array)-1):
        current_marker = "v"
    else:
        current_marker = "."
    plt.scatter([x[0] for x in merkezler_array[i]],[x[1] for x in merkezler_array[i]],
               cmap=cm_bright,
               c=kume_id,s=200,marker=current_marker,edgecolor="k",alpha=0.9)

# çizgiler
for i in range(len(merkezler_array)-1):
    for kume in kume_id:
        plt.plot([merkezler_array[i][kume][0],merkezler_array[i+1][kume][0]],
                [merkezler_array[i][kume][1],merkezler_array[i+1][kume][1]],c=cm_bright.colors[kume])

# Eksen etiketlemeleri
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("Merkezlerin değişimi");