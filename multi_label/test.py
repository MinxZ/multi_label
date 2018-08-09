from __future__ import absolute_import, division, print_function

import json
import multiprocessing as mp
import random

import cv2
import keras
import numpy as np
import pandas as pd
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

data_path = "../data/json"
with open('%s/train.json' % (data_path)) as json_data:
    train = json.load(json_data)
with open('%s/test.json' % (data_path)) as json_data:
    test = json.load(json_data)
with open('%s/validation.json' % (data_path)) as json_data:
    validation = json.load(json_data)

train_img_url = train['images']
train_img_url = pd.DataFrame(train_img_url)
train_ann = train['annotations']
train_ann = pd.DataFrame(train_ann)
train = pd.merge(train_img_url, train_ann, on='imageId', how='inner')

# test data
test = pd.DataFrame(test['images'])

# Validation Data
val_img_url = validation['images']
val_img_url = pd.DataFrame(val_img_url)
val_ann = validation['annotations']
val_ann = pd.DataFrame(val_ann)
validation = pd.merge(val_img_url, val_ann, on='imageId', how='inner')

datas = {'Train': train, 'Test': test, 'Validation': validation}
for data in datas.values():
    data['imageId'] = data['imageId'].astype(np.uint32)


mlb = MultiLabelBinarizer()
train_label = mlb.fit_transform(train['labelId'])

y_test = np.zeros((39706, 228))
x_test = np.arange(y_test.shape[0]) + 1
width = 224


# model_name = 'Xception'
# with CustomObjectScope({'f1_loss': f1_loss, 'f1_score': f1_score, 'precision': precision, 'recall': recall}):
#     model = load_model(f'../models/{model_name}_f1.h5')
# test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# y_pred_test = model.predict_generator(
#     test_datagen.flow(x_test, '../data/test_data', width,
#                       y_test, batch_size=1, shuffle=False),
#     verbose=1)
# np.save(f'../data/json/y_pred_{model_name}', y_pred_test)

y_pred_test_xe_299 = np.load('../data/json/y_pred_Xception299.npy')
# y_pred_test_xe_5945 = np.load('../data/json/y_pred_Xception_5945.npy')
y_pred_test_xe = np.load('../data/json/y_pred_Xception.npy')
# y_pred_test_na = np.load('../data/json/y_pred_NASNetLarge.npy')
# y_pred_test_in = np.load('../data/json/y_pred_InceptionResNetV2.npy')

y_pred_test = (y_pred_test_xe_299 + y_pred_test_xe) / 2
y_pred_test1 = np.round(y_pred_test)
np.sum(y_pred_test1)
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
    file.write(line + '\n')
file.close()


"""
scp z@192.168.3.2:~/data/iM_Fa/data/json/test.csv .
scp ./y_pred_Xception.npy z@192.168.3.2:~/data/iM_Fa/data/json/y_pred_Xception299.npy
"""
