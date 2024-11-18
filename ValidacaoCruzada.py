import pandas as pd 
import tensorflow as tf
import sklearn
import scikeras

pd.__version__, tf.__version__, sklearn.__version__, scikeras.__version__

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential 
from tensorflow.keras import backend as k   

x = pd.read_csv('C:/Users/nanojau/OneDrive/Documentos/CURSOS/DeepLearningWithPythonUdemy/entradas_breast.csv')
y = pd.read_csv('C:/Users/nanojau/OneDrive/Documentos/CURSOS/DeepLearningWithPythonUdemy/saidas_breast.csv')

def criar_rede():
    k.clear_session()
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape=(30,)),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
         tf.keras.layers.Dense(units=16, activation='sigmoide')])
    otimizador = tf.keras.optimizer.Adam(learning_rate = 0.001, clipvalue=0.5)
    rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return rede_neural

    rede_neural = KerasClassifier(model = criar_rede, epochs=100, batch_size=10)

    resultados = cross_val_score(estimator=rede_neural, x = x, y = y, cv = 10, scoring='accuracy')