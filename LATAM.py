# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 22:13:29 2022

@author: Gerardo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('prueba.csv')
X = dataset.iloc[:, [11, 13, 14, 18, 21]].values
X1 = dataset.iloc[:, [11, 13, 14, 18, 21]].values
y = dataset.iloc[:, 20].values

#codificar datos categoricos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

LabelEncoder_X1=LabelEncoder()
X[:, 1] = LabelEncoder_X1.fit_transform(X[:, 1])
LabelEncoder_X2=LabelEncoder()
X[:, 2] = LabelEncoder_X2.fit_transform(X[:, 2])
LabelEncoder_X4=LabelEncoder()
X[:, 4] = LabelEncoder_X4.fit_transform(X[:, 4])

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough')

X = np.array(ct.fit_transform(X), dtype=np.float)
#X = ct.fit_transform(X)
X = X[:, 1:]

#dividir el dataset en conjunto de entrenamiento y conjunto de prueba o testing

from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)

# escalado de variables

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 11, metric = "minkowski", p = 2)
classifier.fit(X_train, y_train)

y_pred  = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



classifier.score(X_test, y_test)


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
# Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
# Recall
from sklearn.metrics import recall_score
recall_score(y_test, y_pred, average=None)
# Precision
from sklearn.metrics import precision_score
precision_score(y_test, y_pred, average=None)

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average=None)
# Method 3: Classification report [BONUS]
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

'''
Usando una SVM

'''
from sklearn.svm import SVC

classifier = SVC(kernel = "sigmoid", random_state = 0)
classifier.fit(X_train, y_train)

y_pred  = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


'''

Bayes

'''

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred  = classifier.predict(X_test)

'''

Arbol de decision (escalado)

'''

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

y_pred  = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

'''

Arbol de decision (SIN escalar)

'''
X1[:, 1] = LabelEncoder_X1.fit_transform(X1[:, 1])
X1[:, 2] = LabelEncoder_X2.fit_transform(X1[:, 2])
X1[:, 4] = LabelEncoder_X4.fit_transform(X1[:, 4])

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough')

X1 = np.array(ct.fit_transform(X1), dtype=np.float)
#X = ct.fit_transform(X)
X1 = X1[:, 1:]

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(X1_train, y1_train)

y_pred  = classifier.predict(X1_test)


cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

'''

Random Forest

'''

from sklearn.ensemble import RandomForestClassifier


classifier = RandomForestClassifier(n_estimators = 15, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

y_pred  = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

'''

ANN

'''

import keras
from keras.models import Sequential
from keras.layers import Dense

    #Inicializar RNA (ANN)
clasificador=Sequential()

    #Añadir capas de entrada y primera capa oculta
clasificador.add(Dense(units=6, kernel_initializer='uniform',
                       activation='relu', input_dim=10))

    #Añadir segunda capa oculta
clasificador.add(Dense(units=6, kernel_initializer='uniform',
                       activation='relu'))

    #Añadir capas de salida
clasificador.add(Dense(units=1, kernel_initializer='uniform',
                       activation='sigmoid'))

    #Compilar RNA o ANN
clasificador.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy'])
'''
            Ajustar el modelo al conjunto de entrenamiento
'''
    # Crear aquí nuestro modelo
clasificador.fit(X_train, y_train, epochs=100, batch_size=10)

    # Predicción de nuestros modelos
y_pred= clasificador.predict(X_test)
y_pred5 = (y_pred>=0.5)
y_pred25 = (y_pred>=0.25)
y_pred24 = (y_pred>=0.24)
y_pred2 = (y_pred>=0.2)

#Elaboraremos matriz de confusión para evaluar el modelo
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred25)

print(classification_report(y_test, y_pred25))

'''

Reducir la dimensión del dataset con ACP

'''

from sklearn.decomposition import PCA

pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

classifier = SVC(kernel = "sigmoid", random_state = 0)
classifier.fit(X_train, y_train)

y_pred  = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()

'''
                        Construir XGBoost (escalado)
'''
#Ajustar el modelo al conjunto de entrenamiento
from xgboost import XGBClassifier
clasificador = XGBClassifier(base_score=0.3, gamma=0.1, max_depth=10, n_estimators=500)
clasificador.fit(X_train, y_train)

# Predicción de nuestros modelos
y_pred= clasificador.predict(X_test)

#Elaboraremos matriz de confusión para evaluar el modelo
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

#Usamos K-Fold Cross Validation
from sklearn.model_selection import cross_val_score as cvs
precisiones=cvs(estimator=clasificador, X=X_train, y=y_train, cv=50)
precisiones.mean()
precisiones.std()

#Mejora Grid Search para optimización del modelo
from sklearn.model_selection import GridSearchCV
parametros=[{'base_score':[0.4, 0.5, 0.6], 'booster':['gbtree'], 'gamma': [0.1, 0.01],
             'max_depth':[3, 6, 8], 'n_estimators':[100, 50, 200]}
    ]
grid_serch=GridSearchCV(estimator=clasificador, 
                        param_grid=parametros,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1)
grid_serch=grid_serch.fit(X_train, y_train)
best_accuracy = grid_serch.best_score_
best_parameters = grid_serch.best_params_


'''
                Reducir la dimensión con LDA
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=1)
X_train=lda.fit_transform(X_train, y_train)
X_test=lda.transform(X_test)

'''
                            FIN
'''
# Crear aquí nuestro modelo

classifier = SVC(kernel = "sigmoid", random_state = 0)
classifier.fit(X_train, y_train)

y_pred  = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))




# Visualización 

'''
                        Construir XGBoost (Sin escalar)
'''

clasificador = XGBClassifier(base_score=0.25, gamma=0.1, max_depth=10, n_estimators=500)
clasificador.fit(X1_train, y1_train)

y_pred= clasificador.predict(X1_test)

cm=confusion_matrix(y1_test, y_pred)

'''

Con modelo de fraudes

'''
from keras.models import Model, load_model
from keras.layers import Input, Dense

dim_entrada = X_train.shape[1]          # 29
capa_entrada = Input(shape=(dim_entrada,))

encoder = Dense(4, activation='tanh')(capa_entrada)
encoder = Dense(2, activation='relu')(encoder)

decoder = Dense(4, activation='tanh')(encoder)
decoder = Dense(10, activation='relu')(decoder)

autoencoder = Model(inputs=capa_entrada, outputs=decoder)

autoencoder.compile(optimizer='sgd', loss='mse')

nits = 100
tam_lote = 32
autoencoder.fit(X_train, X_train, epochs=nits, batch_size=tam_lote, shuffle=True, validation_data=(X_test,X_test), verbose=1)

X_pred = autoencoder.predict(X_test)
ecm = np.mean(np.power(X_test-X_pred,2), axis=1)
print(X_pred.shape)

from sklearn.metrics import precision_recall_curve

precision, recall, umbral = precision_recall_curve(y_test, ecm)


plt.plot(umbral, precision[1:], label="Precision",linewidth=5)
plt.plot(umbral, recall[1:], label="Recall",linewidth=5)
plt.title('Precision y Recall para diferentes umbrales')
plt.xlabel('Umbral')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

y_pred = (ecm>0.7)
y_pred5 = (ecm>=0.5)
y_pred25 = (ecm>=0.25)
y_pred24 = (ecm>=0.24)
y_pred2 = (ecm>=0.2)
y_pred3 = (ecm>=0.3)

cm=confusion_matrix(y_test, y_pred2)

print(classification_report(y_test, y_pred3))

'''

Mezclando ANN y fraudes

'''

ecm2=ecm.reshape(13642,1)
y_tot_p = y_pred + ecm2
y_tot_p_5 = y_tot_p > 0.5
y_tot_p_4 = y_tot_p > 0.4
y_tot_p_44 = y_tot_p > 0.41
cm=confusion_matrix(y_test, y_tot_p_44)
print(classification_report(y_test, y_tot_p_44))