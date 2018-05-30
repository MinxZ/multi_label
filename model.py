from __future__ import absolute_import, division, print_function

import os
from collections import defaultdict

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.applications import Xception
from keras.applications.inception_v3 import preprocess_input
# Legacy functions
from keras.backend.common import (epsilon, floatx, image_data_format,
                                  image_dim_ordering, set_image_dim_ordering)
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *
from keras.utils.generic_utils import has_arg
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import (control_flow_ops, functional_ops,
                                   tensor_array_ops)
from tensorflow.python.training import moving_averages

x_train, y_train, x_val, y_val = load_multi_label_data('../data/json')
t = np.sum(y_train, axis=0)
weight = t / np.max(t)


def binary_crossentropy_weight(target, output, from_logits=False):
    if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))
    return tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=3)


def f1_loss(y_true, y_pred):

    TP = K.sum(y_pred * y_true)
    precision = TP / K.sum(y_pred)
    recall = TP / K.sum(y_true)
    f1 = (1 - 2 * precision * recall / (precision + recall))

    return K.sqrt(f1 + 1)


def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)

    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def precision(y_true, y_pred):
    y_pred = tf.round(y_pred)

    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return precision


def recall(y_true, y_pred):
    y_pred = tf.round(y_pred)

    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return recall


def get_features(MODEL, data, width, batch_size, model_name):
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
    np.save(f'../data/fc_features_{model_name}', features)
    return features


def fc_model(MODEL, x_train, batch_y, width, batch_size, model_name, n_class):
    try:
        features = np.load(f'../data/fc_features_{model_name}.npy')
    except:
        features = get_features(MODEL, x_train, width, batch_size, model_name)

    # Training fc models
    inputs = Input(features.shape[1:])
    x = inputs
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='elu', name='fc')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(n_class, activation='sigmoid', name='predictions')(x)
    model_fc = Model(inputs, x)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(
        filepath=f'../models/fc_{model_name}.h5', verbose=0, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(
        factor=np.sqrt(0.1), patience=5, verbose=2)

    model_fc.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        # loss=f1_loss,
        metrics=[f1_score, precision, recall])
    model_fc.fit(
        features,
        batch_y,
        batch_size=128,
        epochs=10000,
        validation_split=0.1,
        callbacks=[checkpointer, early_stopping])


def build_model(MODEL, width, n_class, model_name, batch_size):
    print(' Build model. \n')
    # Build the model
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights='imagenet', pooling='avg')
    inputs = Input((width, width, 3))
    x = inputs
    # x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='elu', name='fc')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=inputs, outputs=x)

    try:
        print('\n Loading weights. \n')
        model.load_weights(f'../models/fc_{model_name}.h5', by_name=True)
    except:
        print(' Train fc layer firstly.\n')
        try:
            batch_x = np.load('../data/batch_x.npy')
            batch_y = np.load('../data/batch_y.npy')
        except:
            index_array = np.random.permutation(n)[:8192]
            batch_x = np.zeros(
                (len(index_array), width, width, 3),  dtype=np.int8)
            batch_y = y_train[index_array]
            for i, j in enumerate(tqdm(index_array)):
                s_img = cv2.imread(f'../data/train_data/{j+1}.jpg')
                b, g, r = cv2.split(s_img)       # get b,g,r
                rgb_img = cv2.merge([r, g, b])     # switch it to rgb
                x = resizeAndPad(rgb_img, (width, width))
                batch_x[i] = x
            np.save('../data/batch_x', batch_x)
            np.save('../data/batch_y', batch_y)
        fc_model(MODEL, batch_x, batch_y, width,
                 batch_size, model_name, n_class)
        print('\n Loading weights. \n')
        model.load_weights(f'../models/fc_{model_name}.h5', by_name=True)
    return model
