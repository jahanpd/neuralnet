import numpy as np


def import_data(x):
    data = np.genfromtxt(x, delimiter=",")
    y, X = np.hsplit(data, [3])
    trash, y = np.hsplit(y, [2])
    m, n = X.shape
    tr = int(0.6*m)
    X_trn, X_tst = np.vsplit(X, [tr])
    y_trn, y_tst = np.vsplit(y, [tr])
    m_trn, n_trn = X_trn.shape
    m_tst, n_tst = X_tst.shape
    return X_trn, X_tst, y_trn, y_tst, m_trn, n_trn, m_tst, n_tst


def transform_y(y):
    m, n = np.shape(y)
    new_y = np.zeros((m, 2))
    for x in range(m):
        if y[x] == 1:
            new_y[x, 1] = 1
        else:
            new_y[x, 0] = 1
    return new_y


def normalise(x):
    m, n = np.shape(x)
    new_x = np.zeros((m, n))
    mean = np.mean(x, 0)
    std = np.std(x, 0)
    for index, u in np.ndenumerate(x):
        a = index[0]
        b = index[1]
        new_x[a, b] = (u - mean[b])/std[b]
    return new_x
