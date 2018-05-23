from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import random

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm

from image import ImageDataGenerator
from load_data import *
from model import *

# Load datasets
x_train, y_train, x_val, y_val = load_multi_label_data('../data/json')
width = 224
n_class = y_val.shape[1]
n = x_train.shape[0]

# index_array = np.random.permutation(n)[:6000]
# batch_x = np.zeros((len(index_array), width, width, 3))
# batch_y = y_train[index_array]
# for i, j in enumerate(index_array):
#     s_img = cv2.imread(f'../data/train_data/{j+1}.jpg')
#     b, g, r = cv2.split(s_img)       # get b,g,r
#     rgb_img = cv2.merge([r, g, b])     # switch it to rgb
#     x = resizeAndPad(rgb_img, (width, width))
#     batch_x[i] = x

# print(' Train fc layer firstly.\n')
# fc_model(MODEL, batch_x, batch_y, 64)

# Loading model
model_name = 'Xception'
MODEL = Xception
batch_size = 16
model = build_model(MODEL, width, n_class)

# Load weights
print('\n Loading weights. \n')
model.load_weights(f'../models/fc_{model_name}.h5', by_name=True)
print(f' Load fc_{model_name}.h5 successfully.\n')
# model.load_weights('../models/Xception_69_256.h5', by_name=True)

# callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=patience, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(
    filepath=f'../models/{model_name}.h5', verbose=0, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    factor=0.3, patience=reduce_lr_patience, verbose=2)


# datagen and val_datagen
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # preprocessing_function=get_random_eraser(
    #     p=0.2, v_l=0, v_h=255, pixel_level=True),  # 0.1-0.4
    rotation_range=20,  # 10-30
    width_shift_range=0.2,  # 0.1-0.3
    height_shift_range=0.2,  # 0.1-0.3
    shear_range=0.2,  # 0.1-0.3
    zoom_range=0.2,  # 0.1-0.3
    horizontal_flip=True,
    fill_mode='nearest')
# datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Compile model
optimizer = 'SGD'
lr = 5e-3  # 1-5e4
epoch = 1e4
reduce_lr_patience = 5
patience = 10  # reduce_lr_patience+1 + 1


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


print(f"  Optimizer={optimizer} lr={str(lr)} \n")
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    # optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
    # metrics=['binary_accuracy'])
    metrics=[f1_score])

# Start fitting model
print(" Fine tune " + model_name + ": \n")
model.fit_generator(
    datagen.flow(x_train, '../data/train_data', width,
                 y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size / 20,
    validation_data=val_datagen.flow(
        x_val, '../data/val_data', width, y_val, batch_size=batch_size),
    validation_steps=len(x_val) / batch_size,
    epochs=epoch,
    callbacks=[early_stopping, checkpointer, reduce_lr])
