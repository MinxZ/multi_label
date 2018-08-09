import numpy as np


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text
