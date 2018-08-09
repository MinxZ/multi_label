from __future__ import absolute_import, division, print_function

import keras
import numpy as np
import tensorflow as tf
from keras.applications import *
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm


def load_data(model='train', num_fc=3):
    p = np.load('../data/p.npy')
    num = p.shape[0]
    y = {}
    y_train = []
    y_val = []
    y_test = []
    train = p[:int(num * 0.8)]
    val = p[int(num * 0.8):int(num * 0.9)]
    test = p[int(num * 0.9):]
    for i in range(num_fc):
        y[i] = np.load(f'../data/label_{i}.npy')
        y_train.append(y[i][train])
        y_val.append(y[i][val])
        y_test.append(y[i][test])
    x_val = np.load('../data/x_val.npy')
    x_test = np.load('../data/x_test.npy')
    if model == 'train':
        x_train = np.load('../data/x_train.npy')
        return x_train, x_val, x_test, y_train, y_val, y_test
    elif model == 'test':
        return x_val, x_val, x_test, x_val, y_val, y_test
    elif model == 'index':
        return train, val, test, y_train, y_val, y_test


def load_model_config(y_val, model='all'):
    model_config = {
        "Xception": [40, Xception],
        "InceptionResNetV2": [32, InceptionResNetV2],
        "ResNet50": [56, ResNet50],
        "InceptionV3": [56, InceptionV3],
        "DenseNet201": [32, DenseNet201],
        "DenseNet169": [40, DenseNet169],
        "DenseNet121": [56, DenseNet121],
    }
    train, val, test, y_train, y_val, y_test = load_data(model='index')
    if model == 'all':
        fc = [96, 128, 32]
        pred = []
        for i, data in enumerate(y_val):
            pred.append(data.shape[1])
        layer_names = ['category', 'color', 'pattern']
    else:
        fc_model = {'category': 96,
                    'color': 128,
                    'pattern': 32}
        pred_model = {'category': 157,
                      'color': 25,
                      'pattern': 19}
        fc = [fc_model[model]]
        pred = [pred_model[model]]
        layer_names = [model]

    input_shape = (224, 224, 3)
    return model_config, fc, pred, layer_names, input_shape


def tri_fc(inputs, x, fc, pred, layer_names, activation_1='elu', activation_2=['softmax', 'sigmoid', 'sigmoid']):
    num = len(fc)
    processed = {}
    for i in range(num):
        processed[i] = Dropout(0.5)(x)
        processed[i] = Dense(fc[i], activation=activation_1,
                             name=f'{layer_names[i]}_{activation_1}_{fc[i]}')(processed[i])
        processed[i] = Dropout(0.5)(processed[i])
        processed[i] = Dense(pred[i], activation=activation_2[i],
                             name=f'{layer_names[i]}_{activation_2[i]}_{pred[i]}')(processed[i])

        # processed[i] = Dense(fc[i], activation=activation_1,
        #                      name=f'processed{i}_fc{i}')(processed[i])
        # processed[i] = Dropout(0.5)(processed[i])
        # processed[i] = Dense(pred[i], activation=activation_2[i],
        #                      name=layer_names[i])(processed[i])
    if num == 1:
        outputs = processed[0]
    else:
        outputs = [processed[i] for i in range(num)]
    model = Model(inputs, outputs)
    return model


def tri_fc_256(inputs, x, fc, pred, layer_names, activation_1='elu'):
    num = len(fc)
    processed = {}
    for i in range(num):
        processed[i] = Dropout(0.5)(x)
        processed[i] = Dense(fc[i], activation=activation_1,
                             name=f'processed{i}_fc{i}')(processed[i])
    if num == 1:
        outputs = processed[0]
    else:
        outputs = [processed[i] for i in range(num)]
    model = Model(inputs, concatenate(outputs))
    return model


def fc_model_train(x_train_fc, y_train_fc, x_val_fc, y_val_fc, batch_size, cnn_model, fc, pred, layer_names, model_name, preprocess_input, activation_2=['softmax', 'sigmoid', 'sigmoid']):
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='auto')
    multi_fc_model_name = f'{len(fc)}_fc_{model_name}'

    try:
        f_train = np.load(f'../data/f_train_{model_name}.npy')
        f_val = np.load(f'../data/f_val_{model_name}.npy')
        print('Load features successfully.')
    except:
        print('Compute for the features.')
        from keras.preprocessing.image import ImageDataGenerator
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)
        f_train = cnn_model.predict_generator(
            val_datagen.flow(x_train_fc, batch_size=batch_size, shuffle=False),
            steps=len(x_train_fc) / batch_size,
            workers=8,
            use_multiprocessing=True,
            verbose=1)
        np.save(f'../data/f_train_{model_name}', f_train)
        f_val = cnn_model.predict_generator(
            val_datagen.flow(x_val_fc, batch_size=batch_size, shuffle=False),
            steps=len(x_val_fc) / batch_size,
            workers=8,
            use_multiprocessing=True,
            verbose=1)
        np.save(f'../data/f_val_{model_name}', f_val)

    f_input_shape = f_train.shape[1:]
    f_inputs = Input(shape=f_input_shape)
    f_x = f_inputs

    print(f'Train for {len(fc)}_fc model')
    checkpointer = ModelCheckpoint(
        filepath=f'../models/{multi_fc_model_name}.h5', verbose=0, save_best_only=True)
    fc_model = tri_fc(f_inputs, f_x, fc, pred, layer_names)

    losses = {
        f'{layer_names[0]}_{activation_2[0]}_{pred[0]}': "categorical_crossentropy",
        f'{layer_names[1]}_{activation_2[1]}_{pred[1]}': 'binary_crossentropy',
        f'{layer_names[2]}_{activation_2[2]}_{pred[2]}': 'binary_crossentropy'}
    lossWeights = {
        f'{layer_names[0]}_{activation_2[0]}_{pred[0]}': 1,
        f'{layer_names[1]}_{activation_2[1]}_{pred[1]}': 10,
        f'{layer_names[2]}_{activation_2[2]}_{pred[2]}': 10}
    metrics = {
        f'{layer_names[0]}_{activation_2[0]}_{pred[0]}': ["categorical_accuracy"],
        f'{layer_names[1]}_{activation_2[1]}_{pred[1]}': [f1_score],
        f'{layer_names[2]}_{activation_2[2]}_{pred[2]}': [f1_score]}
    opt = 'Adam'
    fc_model.compile(
        optimizer=opt, loss=losses, loss_weights=lossWeights,
        metrics=metrics)

    fc_model.fit(
        f_train,
        y_train_fc,
        validation_data=(f_val, y_val_fc),
        batch_size=64,
        epochs=10000,
        callbacks=[checkpointer, early_stopping])


def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    TP = K.sum(y_pred * y_true)
    precision = TP / K.sum(y_pred)
    recall = TP / K.sum(y_true)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def f1_loss(y_true, y_pred):
    TP = K.sum(y_pred * y_true)
    precision = TP / K.sum(y_pred)
    recall = TP / K.sum(y_true)
    f1 = 2 * precision * recall / (precision + recall)
    f1_loss = 1 - f1
    return f1_loss


def precision(y_true, y_pred):
    y_pred = tf.round(y_pred)
    TP = K.sum(y_pred * y_true)
    precision = TP / K.sum(y_pred)
    return precision


def recall(y_true, y_pred):
    y_pred = tf.round(y_pred)
    TP = K.sum(y_pred * y_true)
    recall = TP / K.sum(y_true)
    return recall
