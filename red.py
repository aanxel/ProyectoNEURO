import data
import keras.models as krm
import keras.optimizers as kro
import keras.layers as krl
import keras.callbacks as krc
import sklearn.metrics as skm
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def build_model(input_shape):
    model = krm.Sequential()
    model.add(
        krl.ConvLSTM2D(filters=64,
                       kernel_size=3,
                       return_sequences=False,
                       data_format="channels_last",
                       input_shape=input_shape[1:]))
    model.add(krl.Dropout(0.2))
    model.add(krl.Flatten())
    model.add(krl.Dense(256, activation="relu"))
    model.add(krl.Dropout(0.3))
    model.add(krl.Dense(input_shape[-1], activation="softmax"))
    return model


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data.load_dataset()
    model = build_model(X_train.shape)
    model.summary()

    opt = kro.SGD(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])

    earlystop = krc.EarlyStopping(patience=7)
    callbacks = [earlystop]

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=40,
                        batch_size=8,
                        shuffle=True,
                        validation_split=0.2,
                        callbacks=callbacks)

    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print(skm.classification_report(y_test, y_pred))
