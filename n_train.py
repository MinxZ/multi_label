from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import random

import cv2
import keras
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

from image import ImageDataGenerator, resizeAndPad
from load_data import *
from model import *

# Load datasets
x_train, y_train, x_val, y_val = load_multi_label_data('../data/json')
width = 331
n_class = y_val.shape[1]
n = x_train.shape[0]

model_name = 'NASNetLarge'
MODEL = NASNetLarge
batch_size = 16

model = build_model(MODEL, width, n_class)
# Load weights
try:
    print('\n Loading weights. \n')
    model.load_weights(f'../models/fc_{model_name}_bc.h5', by_name=True)
except:
    print(' Train fc layer firstly.\n')
    try:
        batch_x = np.load('../data/batch_x.npy')
        batch_y = np.load('../data/batch_y.npy')
    except:
        index_array = np.random.permutation(n)[:8192]
        batch_x = np.zeros((len(index_array), width, width, 3), dtype=int8)
        batch_y = y_train[index_array]
        for i, j in enumerate(tqdm(index_array)):
            s_img = cv2.imread(f'../data/train_data/{j+1}.jpg')
            b, g, r = cv2.split(s_img)       # get b,g,r
            rgb_img = cv2.merge([r, g, b])     # switch it to rgb
            x = resizeAndPad(rgb_img, (width, width))
            batch_x[i] = x
        np.save('../data/batch_x', batch_x)
        np.save('../data/batch_y', batch_y)
    fc_model(MODEL, batch_x, batch_y, width, batch_size, model_name, n_class)
    print('\n Loading weights. \n')
    model.load_weights(f'../models/fc_{model_name}_bc.h5', by_name=True)

# callbacks
reduce_lr_patience = 2
patience = 5  # reduce_lr_patience+1 + 1
early_stopping = EarlyStopping(
    monitor='val_loss', patience=patience, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(
    filepath=f'../models/{model_name}_bc.h5', verbose=0, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    factor=np.sqrt(0.1), patience=reduce_lr_patience, verbose=2)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

lr = 3e-4
optimizer = 'Adam'
print(f"  Optimizer={optimizer} lr={str(lr)} \n")
model.compile(
    # loss=f1_loss,
    loss='binary_crossentropy',
    optimizer=Adam(lr=lr),
    # optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
    metrics=[f1_score, precision, recall])
# Start fitting model
fold = 20
print(" Fine tune " + model_name + ": \n")
epoch = 1e4
model.fit_generator(
    datagen.flow(x_train, '../data/train_data', width,
                 y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size / fold,
    validation_data=val_datagen.flow(
        x_val, '../data/val_data', width, y_val, batch_size=batch_size),
    validation_steps=len(x_val) / batch_size,
    epochs=epoch,
    callbacks=[early_stopping, reduce_lr],
    workers=4)

checkpointer = ModelCheckpoint(
    filepath=f'../models/{model_name}_f1.h5', verbose=0, save_best_only=True)

lr = 1e-4
model.compile(
    loss=f1_loss,
    # loss='binary_crossentropy',
    optimizer=Adam(lr=lr),
    # optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
    metrics=[f1_score, precision, recall])
# Start fitting model
fold = 20
print(" Fine tune " + model_name + ": \n")
epoch = 1e4
model.fit_generator(
    datagen.flow(x_train, '../data/train_data', width,
                 y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size / fold,
    validation_data=val_datagen.flow(
        x_val, '../data/val_data', width, y_val, batch_size=batch_size),
    validation_steps=len(x_val) / batch_size,
    epochs=epoch,
    callbacks=[early_stopping, reduce_lr],
    workers=4)
