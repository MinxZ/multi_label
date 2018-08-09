from collections import defaultdict

import numpy as np

from model import *

# x_train, x_val, x_test, y_train, y_val, y_test = load_data()

y1, y2, y3 = y_train
# yval1, yval2, yval3 = y_val
# y1, y2, y3 = y_test

category_index = defaultdict(list)
for i in range(y1.shape[0]):
    which_c = np.argwhere(y1[i] > 0)[0, 0]
    category_index[which_c].append(i)
color_index = defaultdict(list)
for i in range(y2.shape[0]):
    which_c = np.argwhere(y2[i] > 0)
    if which_c.shape[0] == 1:
        color_index[which_c[0, 0]].append(i)
pattern_index = defaultdict(list)
for i in range(y3.shape[0]):
    which_c = np.argwhere(y3[i] > 0)
    if which_c.shape[0] == 1:
        pattern_index[which_c[0, 0]].append(i)
np.save(f'../data/label_train.npy',
        (category_index, color_index, pattern_index))


# for category, i in category_index.items():
#     print(category, len(i))
for c, i in pattern_index.items():
    print(c, len(i))
for c, i in color_index.items():
    print(c, len(i))


# 7/24

color_index = defaultdict(list)
for i in range(y2.shape[0]):
    which_c = np.argwhere(y2[i] > 0)
    if which_c.shape[0] == 1:
        color_index[which_c[0, 0]].append(i)
    else:
        color_name = []
        for ii in range(which_c.shape[0]):
            color_name.append(which_c[ii, 0])
        color_index[str(color_name)].append(i)

count = 0
for c, i in color_index.items():
    if len(i) > 20:
        print(c, len(i))
        count += 1
print(count)

pattern_index = defaultdict(list)
for i in range(y3.shape[0]):
    which_c = np.argwhere(y3[i] > 0)
    if which_c.shape[0] == 1:
        pattern_index[which_c[0, 0]].append(i)
    else:
        pattern_name = []
        for ii in range(which_c.shape[0]):
            pattern_name.append(which_c[ii, 0])
        pattern_index[str(pattern_name)].append(i)

count = 0
for c, i in pattern_index.items():
    if len(i) > 20:
        print(c, len(i))
        count += 1
print(count)
