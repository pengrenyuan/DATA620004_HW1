import numpy as np


distribution = [
    {'b': [0, 0], 'w': [-1, 1]},
    {'b': [0, 0], 'w': [-1, 1]},
]

def init_parameters_b(layer,dimensions):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer + 1]) * (dist[1] - dist[0]) + dist[0]


def init_parameters_w(layer,dimensions):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer], dimensions[layer + 1]) * (dist[1] - dist[0]) + dist[0]


def init_parameters(dimensions):
    parameter = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in distribution[i].keys():
            if j == 'b':
                layer_parameter['b'] = init_parameters_b(i,dimensions)
                continue
            if j == 'w':
                layer_parameter['w'] = init_parameters_w(i,dimensions)
                continue
        parameter.append(layer_parameter)
    return parameter