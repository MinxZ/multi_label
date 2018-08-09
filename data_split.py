from collections import defaultdict

import numpy as np

data_name = '../data/shoplist_data_0808_size224.npz'
data = np.load(data_name)
y = {}
y[0] = data['category']
y[1] = data['color']
y[2] = data['pattern']


for i in range(3):
    np.save('../data/label_' + str(i), y[i])

num = y[0].shape[0]

try:
    p = np.load(f"../data/{data_name.split('/')[-1]}_p.npy")
    print('Load p.npy successfully')
except:
    p = np.random.permutation(num)
    np.save(f"../data/{data_name.split('/')[-1]}_p", p)
    print('Create indice again.')

images = data['image']
train = p[:int(num * 0.80)]
val = p[int(num * 0.80):int(num * 0.90)]
test = p[int(num * 0.90):]
print("Split for train-val-test on image dataset.")
np.save('../data/x_train', images[train])
np.save('../data/x_val', images[val])
np.save('../data/x_test', images[test])
#
# for x in range(num):
#     np.save('../data/npy/' + str(x), images[x])
#     print(round(x / num * 100, 2))
