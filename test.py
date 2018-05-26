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
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from image import ImageDataGenerator, resizeAndPad
from load_data import *
from model import *


def f1_score_np(y_true, y_pred):
    return 2 * np.sum(np.round(y_pred) * y_true) / (np.sum(y_true) + np.sum(np.round(y_pred)))


y_test = np.zeros((39706, 228))
x_test = np.arange(y_test.shape[0]) + 1
width = 224

model_name = 'Xception_f1_5945'
# model_names = ['Xception_f1_59', 'Xception_f1_5945', 'Xception_f1']
# for model_name in model_names:
with CustomObjectScope({'f1_loss': f1_loss, 'f1_score': f1_score, 'precision': precision, 'recall': recall}):
    model = load_model(f'../models/{model_name}.h5')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
y_pred_test = model.predict_generator(
    test_datagen.flow(x_test, '../data/test_data', width,
                      y_test, batch_size=1, shuffle=False),
    verbose=1)

l = []
y_pred_test1 = np.round(y_pred_test)
where_1 = mlb.inverse_transform(y_pred_test1)

file = open('../data/json/test.csv', 'w')
file.write('image_id,label_id\n')
for i in x_test:
    where_one = where_1[i - 1]
    line = f"{i},"
    for x in where_one:
        line += f'{x} '
    if line[-1] == ' ':
        line = line[:-1]
    l.append(line)
    file.write(line + '\n')
file.close()

y_pred_test4 = (y_pred_test3 + y_pred_test) / 2
print(f1_score_np(y_pred_test3, y_pred_test))
print(f1_score_np(y_pred_test4, y_pred_test))
print(f1_score_np(y_pred_test4, y_pred_test3))


val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
y_pred_val = model.predict_generator(
    val_datagen.flow(x_val, '../data/val_data', width,
                     y_val, batch_size=1, shuffle=False),
    verbose=1)

ll = []
y_pred_val1 = np.round(y_pred_val)
where_1 = mlb.inverse_transform(y_pred_val1)

file = open('../data/json/val.csv', 'w')
file.write('image_id,label_id\n')
for i in x_val:
    where_one = where_1[i - 1]
    line = f"{i},"
    for x in where_one:
        line += f'{x} '
    ll.append(line)
    file.write(line[:-1] + '\n')
file.close()

y_pred_val4 = (y_pred_val3 + y_pred_val) / 2
print(f1_score_np(y_val, y_pred_val3))
print(f1_score_np(y_val, y_pred_val))
print(f1_score_np(y_val, y_pred_val4))


# y_pred_test3 = y_pred_test
# l3 = l
#
# y_pred_val3 = y_pred_val
# ll3 = ll
"""
scp z@192.168.3.2:~/data/iM_Fa/data/json/test.csv .
"""
