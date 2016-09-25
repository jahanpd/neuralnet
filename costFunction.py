import numpy as np
from sigmoid import sigmoid


def costFunction(X_trn, y_trn, Th1, Th2, Th3, Th4, ThOut, m_trn, n_trn):
    X_trn = np.insert(X_trn, [0], np.ones((m_trn, 1)), axis=1)
    h0 = np.dot(X_trn, Th1)
    h0 = sigmoid(h0)
    h0 = np.insert(h0, [0], np.ones((m_trn, 1)), axis=1)
    h0 = np.dot(h0, Th2)
    h0 = sigmoid(h0)
    h0 = np.insert(h0, [0], np.ones((m_trn, 1)), axis=1)
    h0 = np.dot(h0, Th3)
    h0 = sigmoid(h0)
    h0 = np.insert(h0, [0], np.ones((m_trn, 1)), axis=1)
    h0 = np.dot(h0, Th4)
    h0 = sigmoid(h0)
    h0 = np.insert(h0, [0], np.ones((m_trn, 1)), axis=1)
    h0 = np.dot(h0, ThOut)
    h0 = sigmoid(h0)
    print(h0)
    print(1-y_trn)
    Jpart1 = np.dot(y_trn, (np.log(h0).T))
    print(Jpart1)
    Jpart2 = np.dot((1-y_trn), (np.log((1.00001-h0))).T)
    print(Jpart2)
    J = Jpart1 + Jpart2
    J = np.sum(J)
    print(J)
