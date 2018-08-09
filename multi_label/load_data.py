import numpy as np


def load_multi_label_data(dir='../data'):
    print('\n Loading Datasets. \n')
    y_val = np.load(f'{dir}/val.npy')
    x_val = np.arange(y_val.shape[0]) + 1
    y_train = np.load(f'{dir}/train.npy')
    x_train = np.arange(y_train.shape[0]) + 1
    return x_train, y_train, x_val, y_val
