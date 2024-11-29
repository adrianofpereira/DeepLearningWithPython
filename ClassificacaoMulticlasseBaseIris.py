import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn

pd.__version__, np.__version__, tf.__version__, sklearn.__version__

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils as np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

base = pd.read_csv('C:/Users/nanojau/OneDrive/Documentos/CURSOS/DeepLearningWithPythonUdemy/iris.csv')

x = base.iloc[:, 0:4].values
x

y = base.iloc[:, 4].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y

y = np_utils.to_categorical(y)
y

y.shape

x_treinamento, x_teste, y_treinamento, y_teste= train_test_split(x, y, test_size=0.25) 

x_treinamento.shape, x_teste.shape

y_treinamento.shape, y_teste.shape

rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape = (4,)),
    tf.keras.layers.Dense(units = 4, activation = 'relu'),
    tf.keras.layers.Dense(units = 4, activation = 'relu'),
    tf.keras.layers.Dense(units = 3, activation = 'softmax')
])

rede_neural.summary()

rede_neural.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

rede_neural.fit(x_treinamento, y_treinamento, batch_size = 10, epochs = 1000)

rede_neural.evaluate(x_teste, y_teste)

previsoes = rede_neural.predict(x_teste)
print(previsoes)

previsoes = previsoes > 0.5
print(previsoes)

y_teste2 = [np.argmax(t) for t in y_teste]
print(y_teste2)

previsoes2 = [np.argmax(t) for t in previsoes]
print(previsoes2)

from sklearn.metrics import accuracy_score

accuracy_score(y_teste2, previsoes2)

# 0 setosa, 1 versicolor, 2 virginica

confusion_matrix(y_teste2, previsoes2)