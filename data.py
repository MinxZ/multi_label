import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

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
validation_label = mlb.transform(validation['labelId'])
dummy_label_col = list(mlb.classes_)
print(dummy_label_col)

for data in [validation_label, train_label, test]:
    print(data.shape)

np.save(f'{data_path}/train', train_label)
np.save(f'{data_path}/val', validation_label)
