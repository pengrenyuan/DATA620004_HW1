import numpy as np


def relu(x):
    return np.where(x >= 0, x, 0)


def d_relu(x):
    return np.where(x >= 0, 1, 0)


def softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


def d_softmax(x):
    sm = softmax(x)
    return np.diag(sm) - np.outer(sm, sm)

activation = [relu, softmax]
differential = {softmax: d_softmax, relu: d_relu}