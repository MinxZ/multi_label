model_name = 'Xception_f1_5945'
with CustomObjectScope({'f1_loss': f1_loss, 'f1_score': f1_score, 'precision': precision, 'recall': recall}):
    model = load_model(f'../models/{model_name}.h5')

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
y_pred_val = model.predict_generator(
    val_datagen.flow(x_val, '../data/val_data', width,
                     y_val, batch_size=1, shuffle=False),
    verbose=1)

y_pred_val1 = np.round(y_pred_val)
where_1 = mlb.inverse_transform(y_pred_val1)

file = open('../data/json/val.csv', 'w')
file.write('image_id,label_id\n')
for i in x_val:
    where_one = where_1[i - 1]
    line = f"{i},"
    for x in where_one:
        line += f'{x} '
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
