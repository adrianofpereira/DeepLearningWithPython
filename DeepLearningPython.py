import pandas as pd

x = pd.read_csv('C:/Users/nanojau/OneDrive/Documentos/CURSOS/DeepLearningWithPythonUdemy/entradas_breast.csv')
y = pd.read_csv('C:/Users/nanojau/OneDrive/Documentos/CURSOS/DeepLearningWithPythonUdemy/saidas_breast.csv')


import sklearn
from sklearn.model_selection import train_test_split

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.25)

x_treinamento.shape, y_treinamento.shape

x_teste.shape, y_teste.shape

import tensorflow as tf
from tensorflow.keras.models import Sequential

tf.__version__

rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape =(30,)), ##shape será a quantidade de neurônios utilizados na entrada
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),    ##Na camada oculta utilizamos o Dense que é o neurônio de uma camada conectado a todos os neurônios de outra camada
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
])

rede_neural.summary()

otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)

rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

rede_neural.fit(x_treinamento, y_treinamento, batch_size = 10, epochs = 100)

pesos0 = rede_neural.layers[0].get_weights()
pesos0

len(pesos0)

len(pesos0[1])

pesos1 = rede_neural.layert[1].get_weights
pesos1

pesos2 = rede_neural.layert[2].get_weights
pesos2

previsoes = rede_neural.predict(x_teste)

previsoes = previsoes > 0.5

previsoes

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_teste,previsoes)

confusion_matrix(y_teste,previsoes)

44+85,9+5

129/143, 15/143

rede_neural.evaluate(x_teste, y_teste)