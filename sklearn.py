from sklearn import datasets # librairie pour la dataset
from sklearn.model_selection import train_test_split  # prendre un % train et % test
import sklearn.metrics as sk # metrics contient r2_score,mean_squared_error,classification_report
from sklearn import neighbors

import pandas as pd

import numpy as np

iris=datasets.load_iris()

x=iris.data
y=iris.target
data=[0 for i in range( len(y))]
for i in range (len(y)):
   data[i]=x[i].tolist () + [y[i]]

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.5)




############
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

###########

from sklearn.preprocessing import LabelEncoder

data_car=pd.read_csv('car.DATA')
data_car.columns = ['buying','maint','doors','persons','lug_boot','safety','class']#ajouter les titres des coloumns
Y=data_car['class']
X=data_car.loc[:, data_car.columns != 'class'].values #.values == to array
#X=data_car[["buying","maint","safety"]].values
####1er methode####
Le=LabelEncoder() # ya9bal fa9At array!!!!!!!!
for i in range(len(X[0])):
    X[:,i]=Le.fit_transform(X[:,i])
####2eme methode####
label_mapping={'unacc':0,'acc':1,'good':2,'vgood':3}
Y=data_car['class'].map(label_mapping) # blasst hadi hatti hadi
Y=np.array(Y)


knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')# uniform== effet égale pour tout les voisins, distance== valorise les voisins proches
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)
score=sk.r2_score(Y_test,Y_pred)
MSE=sk.mean_squared_error(Y_test,Y_pred)
accuracy=sk.accuracy_score(Y_test,Y_pred)



######  SVM ########

from sklearn import datasets # librairie pour la dataset
from sklearn.model_selection import train_test_split  # prendre un % train et % test
import sklearn.metrics as sk
from sklearn import svm

iris=datasets.load_iris()
X=iris.data
Y=iris.target
classes=iris.target_names.tolist()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

model=svm.SVC()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
erreur=sk.mean_absolute_error(Y_test,Y_pred)
score=sk.accuracy_score(Y_test,Y_pred)

for i in range(len(Y_pred)):
   print(classes[Y_pred[i]])
