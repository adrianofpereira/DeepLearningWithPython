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