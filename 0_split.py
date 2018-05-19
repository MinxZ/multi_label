from collections import defaultdict

import numpy as np

dir = '../data'

for train_test in ['train', 'test']:
    # Loading Datasets
    print('Loading Datasets. \n')
    labels = np.load(f'{dir}/labels_{train_test}.npy')

    category_index = defaultdict(list)
    for i, label in enumerate(labels):
        category = f"{label[2]}"
        category_index[category].append(i)

    color_index = defaultdict(list)
    for i, label in enumerate(labels):
        color = f"{label[0]}"
        color_index[color].append(i)

    pattern_index = defaultdict(list)
    for i, label in enumerate(labels):
        pattern = f"{label[1]}"
        pattern_index[pattern].append(i)

    label_all = (category_index, color_index, pattern_index)
    print(f'label_{train_test} \n')
    np.save(f'{dir}/label_{train_test}', label_all)

for train_test in ['train', 'test']:
    # Loading Datasets
    print('Loading Datasets. \n')
    labels = np.load(f'{dir}/labels_{train_test}.npy')

    category_index = defaultdict(list)
    for i, label in enumerate(labels):
        category = f"{label[2]}"
        category_index[category].append(i)

    color_index = defaultdict(list)
    for i, label in enumerate(labels):
        color = f"{label[0]}"
        color_index[color].append(i)

    pattern_index = defaultdict(list)
    for i, label in enumerate(labels):
        pattern = f"{label[1]}"
        pattern_index[pattern].append(i)

    label_all = (category_index, color_index, pattern_index)
    print(f'label_{train_test} \n')
    np.save(f'{dir}/label_{train_test}', label_all)
