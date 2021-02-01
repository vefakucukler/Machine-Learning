# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 03:16:07 2019

@author: vefa
"""

# Numpy nümerik hesap kütüphanesi
import numpy as np

# Matplotlib grafik çizim kütüphanesi
import matplotlib.pyplot as plt
# Seaborn dark theme iyidir :)
import seaborn as sns
sns.set_style("darkgrid")

# Özel paketimizden de biraz ekleyelim
import sys
sys.path.append("..")
from fonk import generate_noisy_line
from sklearn.decomposition import PCA


# gürültülü bir çizgi yaratalım
x,y=generate_noisy_line(y_noise=5,x=np.linspace(-10,10,500),random_seed=42)
# x ve y değişkenlerini bütünleşik vektör haline getiriyoruz
X = np.vstack([x,y]).T

# PCA öncesi veriyi normalize etmemiz lazım
# Bunun için ortalamayı çıkarıp standart sapmaya bölüyoruz
# axis=0 yapmamızın sebebi 2 boyutlu bir vektör elde etmek
X = (X - X.mean(axis=0))/X.std(axis=0)

plt.figure()
# Şimdi bunları basitçe çizdirelim
plt.scatter(X[:,0], X[:,1])
plt.xlabel("x değişkeni")
plt.ylabel("y değişkeni")

# x=0 ve y=0 eksenlerini çizelim
plt.axhline(0, color='black')
plt.axvline(0, color='black');

pca = PCA(n_components = 2)
pca.fit(X);

print(pca.components_)
print(pca.explained_variance_)

plt.figure()
plt.bar([1,2],pca.explained_variance_ratio_,width=0.2,color="paleturquoise",edgecolor="indigo")
plt.plot([1,2],np.cumsum(pca.explained_variance_ratio_),'r')
plt.text(1,pca.explained_variance_ratio_[0],s="%.2f" % pca.explained_variance_ratio_[0]);
plt.text(2,pca.explained_variance_ratio_[1],s="%.2f" % pca.explained_variance_ratio_[1]);
plt.xlabel("PC numarası")
plt.ylabel("Varyans")
plt.legend(["Toplam varyans (oransal)","Varyans (oransal)"]);
plt.figure()
plt.scatter(X[:,0], X[:,1],alpha=0.2)
plt.xlabel("x değişkeni")
plt.ylabel("y değişkeni")

# x=0 ve y=0 eksenlerini çizelim
plt.axhline(0, color='black',ls="-.",alpha=0.3)
plt.axvline(0, color='black',ls="-.",alpha=0.3);

# Anotasyon kullanarak "principal component"leriçizdirelim
v0 = pca.mean_
v1 = pca.mean_ + (pca.components_[0])*(np.sqrt(pca.explained_variance_[0])*2)
arrowprops=dict(arrowstyle='<-',
                linewidth=2,
               shrinkA=0, shrinkB=0,color='r');
plt.annotate('', v0, v1, arrowprops=arrowprops);
plt.text(v1[0],v1[1],"PC2")

v0 = pca.mean_
v1 = pca.mean_ + (pca.components_[1])*(np.sqrt(pca.explained_variance_[1])*2)
arrowprops=dict(arrowstyle='<-',
                linewidth=2,shrinkA=0, shrinkB=0,color='b')
plt.annotate('', v0, v1, arrowprops=arrowprops);
plt.text(v1[0],v1[1],"PC1")

# Eksenleri eşit göstermemiz önemli !!!
# Yoksa PC'lerin birbirine dik olduğunu şekilden anlayamazsınız !!!
plt.axis("equal");
np.dot(pca.components_[1],pca.components_[0])

X_new = pca.transform(X)
pc_new = pca.transform(pca.components_)

plt.figure()
plt.scatter(X_new[:,0], X_new[:,1],alpha=0.2)
plt.xlabel("PC1")
plt.ylabel("PC2")

mean_new = X_new.mean(axis=0)

# Anotasyon kullanarak "principal component"leriçizdirelim
v0 = mean_new
v1 = mean_new + pc_new[0]
arrowprops=dict(arrowstyle='<-',
                linewidth=2,
               shrinkA=0, shrinkB=0,color='r');
plt.annotate('', v0, v1, arrowprops=arrowprops);
plt.text(v1[0],v1[1],"PC1")

v0 = mean_new
v1 = np.round(mean_new + pc_new[1])

arrowprops=dict(arrowstyle='<-',
                linewidth=2,color='b')
plt.annotate('', v0, v1, arrowprops=arrowprops,);
plt.text(v1[0],v1[1],"PC2")

# Eksenleri eşit göstermemiz önemli !!!
# Yoksa PC'lerin birbirine dik olduğunu şekilden anlayamazsınız !!!
plt.axis("equal");

plt.figure()
plt.scatter(X_new[:,0], X_new[:,1],alpha=0.2)
# ilk eksende PC1 olduğu için
plt.scatter(X_new[:,0],np.zeros(X_new[:,0].shape),alpha=0.1,color='r')

# Projeksiyon çizgilerini de çizdirelim
for i in range(len(X_new[:,0])):
    plt.plot([X_new[i,0],X_new[i,0]],[X_new[i,1],0],color="deeppink",alpha=0.1)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.axis("equal");
from sklearn.cluster.k_means_ import KMeans

kmeans=KMeans(n_clusters=3)
kmeans.fit(x,y)
print(kmeans.cluster_centers_)
print(pd.crosstab(y,kmeans.labels_))