from deepsvdd import DeepSVDD, Objectives
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import keras


REPRESENTATION_DIM = 32

# MNIST data
cls = 1
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

mask = y_train == cls

x_train = x_train[mask]
x_train = np.expand_dims(x_train / 255., axis=-1).astype(np.float32)
x_test = np.expand_dims(x_test / 255., axis=-1).astype(np.float32)

y_test = (y_test == cls).astype(np.float32)


# Network
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(8, (5, 5), padding='same', use_bias=False, input_shape=(28, 28, 1)))
model.add(keras.layers.LeakyReLU(1e-2))
model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
model.add(keras.layers.MaxPool2D())

model.add(keras.layers.Conv2D(4, (5, 5), padding='same', use_bias=False))
model.add(keras.layers.LeakyReLU(1e-2))
model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
model.add(keras.layers.MaxPool2D())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(REPRESENTATION_DIM, use_bias=False))


# Train and test
svdd = DeepSVDD(model, representation_dim=REPRESENTATION_DIM, objective=Objectives.ONE_CLASS)
svdd.fit(x_train, x_test, y_test, 'mnist_test', n_epochs=50, verbose=True)
score = svdd.predict(x_test)
auc = roc_auc_score(y_test, -score)
print(f'Final ROC AUC score = {auc}')
