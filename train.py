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

from image import ImageDataGenerator, resizeAndPad
from load_data import *

# from model import *

# Load datasets
x_train, y_train, x_val, y_val = load_multi_label_data('../data/json')
width = 224
n_class = y_val.shape[1]
n = x_train.shape[0]

model_name = 'Xception'
with CustomObjectScope({'f1_loss': f1_loss, 'f1_score': f1_score, 'precision': precision, 'recall': recall}):
    model = load_model(f'../models/Xception_f1_59.h5')


a = 0
if a:
    MODEL = Xception
    model = build_model(MODEL, width, n_class)

    # print(' Train fc layer firstly.\n')
    # fc_model(MODEL, batch_x, batch_y, width, 64)

    # Load weights
    print('\n Loading weights. \n')
    model.load_weights(f'../models/fc_{model_name}.h5', by_name=True)
    # model.load_weights(f'../models/{model_name}_{loss_name}.h5', by_name=True)

# callbacks
reduce_lr_patience = 5
patience = 10  # reduce_lr_patience+1 + 1
early_stopping = EarlyStopping(
    monitor='val_loss', patience=patience, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(
    filepath=f'../models/{model_name}.h5', verbose=0, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    factor=np.sqrt(0.1), patience=reduce_lr_patience, verbose=2)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

b = 0
if b:
    # Compile model
    optimizer = 'Adam'
    lr = 1e-6  # 1-5e4
    print(f"  Optimizer={optimizer} lr={str(lr)} \n")
    model.compile(
        loss=f1_loss,
        # loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        # optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
        metrics=[f1_score, precision, recall])


# Start fitting model
fold = 100
print(" Fine tune " + model_name + ": \n")
batch_size = 16
epoch = 1e4
model.fit_generator(
    datagen.flow(x_train, '../data/train_data', width,
                 y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size / fold,
    validation_data=val_datagen.flow(
        x_val, '../data/val_data', width, y_val, batch_size=batch_size),
    validation_steps=len(x_val) / batch_size,
    epochs=epoch,
    callbacks=[early_stopping, checkpointer, reduce_lr],
    workers=4)
