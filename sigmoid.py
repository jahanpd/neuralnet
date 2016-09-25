import numpy as np


def sigmoid(z):
    h0 = 1/(1+np.exp(z))
    return h0
