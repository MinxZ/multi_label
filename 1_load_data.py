import numpy as np

from tools import unison_shuffled_copies


def load_multi_label_data(dir='../data'):
    print('\n Loading Datasets. \n')
    try:
        y_val = np.load(f'{dir}/y_val.npy')
        y_train = np.load(f'{dir}/y_train.npy')
    except:

        print('Train val split again.')
        y = np.load(f'{dir}/y.npy')
        X = np.arange(y.shape[0] + 1) + 1

        X, y = unison_shuffled_copies(X, y)
        dvi = int(X.shape[0] * 0.9)
        x_train = X[:dvi, :, :, :]
        y_train = y[:dvi, :]
        x_val = X[dvi:, :, :, :]
        y_val = y[dvi:, :]

        print('Saving data.')
        np.save(f'{dir}/y_val', y_val)
        np.save(f'{dir}/x_val', x_val)
        np.save(f'{dir}/y_train', y_train)
        np.save(f'{dir}/x_train', x_train)
    return x_train, y_train, x_val, y_val
