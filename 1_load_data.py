import numpy as np

from tools import unison_shuffled_copies


def load_multi_label_data(dir='../data'):
    print('\n Loading Datasets. \n')
    try:
        y_val = np.load(f'{dir}/y_val.npy')
        x_val = np.load(f'{dir}/x_val.npy')
        y_train = np.load(f'{dir}/y_train.npy')
        x_train = np.load(f'{dir}/x_train.npy')
    except:
        print('Train val split again.')
        X = np.load(f'{dir}/X_train.npy')
        n = X.shape[0]
        width = X.shape[1]
        (category_index, color_index, pattern_index) = np.load(
            f'{dir}/label_train.npy')
        n_class = len(category_index) + len(color_index) + len(pattern_index)
        y = np.zeros((n, n_class), dtype=np.uint8)

        key = -1
        for index in [category_index, color_index, pattern_index]:
            for class_name, indexes in sorted(index.items(), key=lambda x: x[0]):
                key += 1
                for i in indexes:
                    y[i][key] = 1

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


def load_data(dir='../data'):
    print('\n Loading Datasets. \n')
    try:
        y_val = np.load(f'{dir}/y_val.npy')
        x_val = np.load(f'{dir}/x_val.npy')
        y_train = np.load(f'{dir}/y_train.npy')
        x_train = np.load(f'{dir}/x_train.npy')
    except:
        print('Train val split again.')
        X = np.load(f'{dir}/X_train.npy')
        n = X.shape[0]
        width = X.shape[1]
        try:
            (category_index, class_index, class_category) = np.load(
                f'{dir}/class_train.npy')
            n_class = len(class_index)
            y = np.zeros((n, n_class), dtype=np.uint8)
            key = -1
            for class_name, indexes in sorted(class_index.items(), key=lambda x: x[0]):
                key += 1
                for i in indexes:
                    y[i][key] = 1
        except:
            y = np.load(f'y.npy')

        X, y = unison_shuffled_copies(X, y)
        dvi = int(X.shape[0] * 0.9)
        x_train = X[:dvi, :, :, :]
        y_train = y[:dvi, :]
        x_val = X[dvi:, :, :, :]
        y_val = y[dvi:, :]

        np.save(f'{dir}/y_val', y_val)
        np.save(f'{dir}/x_val', x_val)
        np.save(f'{dir}/y_train', y_train)
        np.save(f'{dir}/x_train', x_train)
    return x_train, y_train, x_val, y_val
