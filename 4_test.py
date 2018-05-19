def binary_accuracy(y_true, y_pred):
    return np.mean(np.equal(y_true, np.round(y_pred)))


x = Lambda(preprocess_input, name='preprocessing')(x)
y_pred = model.predict(x_train, verbose=1)
acc = binary_accuracy(y_train, y_pred)
print(acc)

y_pred_val = model.predict(x_val, verbose=1)
acc_val = binary_accuracy(y_val, y_pred_val)
print(acc_val)

cd / mnt / s3 / keras / multi_label / train/
