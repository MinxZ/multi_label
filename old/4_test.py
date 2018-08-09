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
from model import *


def f1_score_np(y_true, y_pred):
    return 2 * np.sum(np.round(y_pred) * np.round(y_true)) / (np.sum(np.round(y_true)) + np.sum(np.round(y_pred)))


index_array = np.arange(y_val.shape[0])
batch_x = np.zeros((len(index_array), width, width, 3))
batch_y = y_val
for i, j in enumerate(tqdm(index_array)):
    s_img = cv2.imread(f'../data/val_data/{j+1}.jpg')
    b, g, r = cv2.split(s_img)       # get b,g,r
    rgb_img = cv2.merge([r, g, b])     # switch it to rgb
    x = resizeAndPad(rgb_img, (width, width))
    batch_x[i] = x

model_names = ['Xception_f1_59', 'Xception_f1_5945']
for model_name in model_names:
    with CustomObjectScope({'f1_loss': f1_loss, 'f1_score': f1_score, 'precision': precision, 'recall': recall}):
        model = load_model(f'../models/{model_name}.h5')

    # y_pred_val = model.predict(batch_x, verbose=1)
    # print(model_name, f1_score(y_val, y_pred_val))

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    y_pred_val = model.predict_generator(
        val_datagen.flow(x_val, '../data/val_data', width,
                         y_val, batch_size=3, shuffle=False),
        verbose=1)
    print(model_name, f1_score_np(y_val, y_pred_val))
