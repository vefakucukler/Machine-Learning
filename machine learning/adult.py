import pandas
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster.k_means_ import KMeans
from sklearn.decomposition import PCA
import pandas as pd
# Load dataset
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
dataset = pandas.read_csv('adult.data', names=names)


# Split dataset
array = dataset.values
X = array[:,0:1]+array[:,2:3]+array[:,4:5]+array[:,10:13] #bağımlı değişkenler AGE -> 0:1 fnlwgt-> 2:3
#print(X)

Y = array[:,14] # income -> 14 
#print(Y)


# Veri kümesinin eğitim ve test verileri olarak ayrılması
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=8)

models = [
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('DT', DecisionTreeClassifier()),
    ('NB', GaussianNB())
]

# Modeller için 'cross validation' sonuçlarının  yazdırılması
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
   
    
print('***********NB**************')  
nb = LogisticRegression()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print('***********DT**************')  
dt = LogisticRegression()
dt.fit(X_train, Y_train)
predictions = dt.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


print('***********LDA**************')  
lda = LogisticRegression()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print('***********LR**************')  
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print('***********KNN**************')  
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print('***********PCA**************')

pca = PCA()
X_train1 = pca.fit_transform(X_train)
X_validation1 = pca.transform(X_validation)
#print(pca.explained_variance_ratio_)  #her bileşen için varyans değerlerni gösteriyor
print('************************')
print("\n\n",X_train1)
print('*******KMEANS*************')

kmeans = KMeans(n_clusters=5)
kmeans.fit(X,Y)
#print(kmeans.cluster_centers_)
print(pd.crosstab(Y,kmeans.labels_))
