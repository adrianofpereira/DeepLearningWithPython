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
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
])

rede_neural.summary()

rede_neural.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

rede_neural.fit(x_treinamento, y_treinamento, batch_size = 10, epochs = 100)