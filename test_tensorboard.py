from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.metrics import (CategoricalAccuracy)
import tensorflow as tf
import shutil

# Data creation
dataset = load_digits()['data']
X, y = dataset[:, :-1], dataset[:, -1]
n_atrs = X.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model creation
model = Sequential()
model.add(Dense(20, activation='sigmoid', input_dim=n_atrs))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=SGD(),
              loss='categorical_crossentropy',
              metrics=[CategoricalAccuracy(), 'mean_squared_error'])

log_dir = "logs/fit/"
shutil.rmtree('./' + log_dir, ignore_errors=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=5,
                                                      update_freq=5,
                                                      write_images=True)
model.fit(X_train,
          y_train,
          epochs=100,
          validation_split=0.20,
          callbacks=[tensorboard_callback],
          verbose=0)
model.evaluate(X_test, y_test)
