import numpy as np


def sigmoid(z):
    exp = np.exp(z)
    den=1+exp
    h0 = 1/den
    return h0
