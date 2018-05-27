def f1_loss(y_true, y_pred):

    TP = K.sum(y_pred * y_true)
    precision = TP / K.sum(y_true)
    recall = TP / K.sum(y_pred)
    # recall = TP / K.sum(K.round(y_pred))
    f1 = (1 - 2 * precision * recall / (precision + recall))

    return f1


def f1_loss(y_true, y_pred):
    return 1 - 2 * K.sum(y_pred * y_true) / (K.cast(K.sum(y_true), tf.float32) + K.sum(y_pred))


def f1_loss_log(y_true, y_pred):
    TP = K.sum(K.binary_crossentropy(y_true, y_pred))
    # return TP / (K.cast(K.sum(y_true), tf.float32) + K.sum(K.round(y_pred))
    return TP / (K.cast(K.sum(y_true), tf.float32) + K.sum(y_pred))


def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)

    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def f1_loss_log2(y_true, y_pred):
    return (K.sum(-K.log(y_pred) * y_true - K.log(1 - y_pred) * (1 - y_true))) / (K.cast(K.sum(y_true), tf.float32) + K.sum(K.round(y_pred)))


def f1_score(y_true, y_pred):
    return 2 * np.sum(np.round(y_pred) * y_true) / (np.sum(y_true) + np.sum(np.round(y_pred)))


def f1_loss(y_true, y_pred):
    return 1 - 2 * np.sum(y_pred * y_true) / (np.sum(y_true) + np.sum(y_pred))


def f1_loss_round(y_true, y_pred):
    return 1 - 2 * np.sum(y_pred * y_true) / (np.sum(y_true) + np.sum(np.round(y_pred)))


def f1_log(y_true, y_pred):
    return -(np.sum(np.log(y_pred) * y_true + np.log(1 - y_pred) * (1 - y_true))) / (np.sum(y_true) + np.sum(y_pred))


def f1_log_round(y_true, y_pred):
    return -(np.sum(np.log(y_pred) * y_true + np.log(1 - y_pred) * (1 - y_true))) / (np.sum(y_true) + np.sum(np.round(y_pred)))


y_pred = np.array([0.9, 0.9, 0.9, 0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                   0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, ])
y_true = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])
print(f1_loss(y_true, y_pred))
print(f1_loss_round(y_true, y_pred))
print(f1_log(y_true, y_pred))
print(f1_log_round(y_true, y_pred))
print(f1_score(y_true, y_pred))

- (np.sum(np.log(y_pred) * y_true + np.log(1 - y_pred) * (1 - y_true)))


y_pred = np.array([0.4, 0.95, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                   0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, ])
y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])
print(f1_loss(y_true, y_pred))
print(f1_score(y_true, y_pred))
print(precision(y_true, y_pred))
print(recall(y_true, y_pred))
binary_crossentropy(y_true, y_pred)
- (np.sum(np.log(y_pred) * y_true + np.log(1 - y_pred) * (1 - y_true)))


- 2 * np.sum(y_pred * y_true) + (np.sum(y_true) + np.sum(np.round(y_pred)))
- 2 * np.sum(y_pred * y_true) + (np.sum(y_true) + np.sum(y_pred))


def f1_score(y_true, y_pred):
    return 1 - 2 * np.sum(np.round(y_pred) * y_true) / (np.sum(y_true) + np.sum(y_pred))


print(f1_score(y_true, y_pred))
