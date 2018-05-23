import numpy as np
from keras.applications import Xception
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *


def get_features(MODEL, data, batch_size):
    cnn_model = MODEL(input_shape=(width, width, 3),
                      include_top=False,  weights='imagenet', pooling='avg')
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='elu', name='fc')(x)
    cnn_model = Model(inputs, x)
    # cnn_model.load_weights('../models/Xception_69_256.h5', by_name=True)

    features = cnn_model.predict(data, batch_size=batch_size, verbose=1)
    np.save('../data/fc_features', features)
    return features


def fc_model(MODEL, x_train, y_train, batch_size):
    try:
        features = np.load('../data/fc_features.npy')
    except:
        features = get_features(MODEL, x_train, 16)

    # Training fc models
    inputs = Input(features.shape[1:])
    x = inputs
    x = Dropout(0.5)(x)
    x = Dense(256, activation='elu', name='fc')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='sigmoid', name='predictions')(x)
    model_fc = Model(inputs, x)

    model_fc.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'])
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(
        filepath=f'../models/fc_{model_name}.h5', verbose=0, save_best_only=True)
    model_fc.fit(
        features,
        y_train,
        batch_size=128,
        epochs=10000,
        validation_split=0.1,
        callbacks=[checkpointer, early_stopping])


def build_model(MODEL, width, n_class):
    print(' Build model. \n')
    # Build the model
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights='imagenet', pooling='avg')
    inputs = Input((width, width, 3))
    x = inputs
    # x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='elu', name='fc')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=inputs, outputs=x)

    return model
