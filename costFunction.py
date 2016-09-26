import numpy as np
from sympy import *
import mpmath as mp

# sympy elementwise functions

sig = lambda x: 1.0/(1.0+mp.exp(x))
mlog = lambda x: mp.log(x)
sub1 = lambda x: 1.0-x
sqar = lambda x: mp.power(x, 2)


def costFunction(X_trn, y_trn, Th1, Th2, Th3, Th4, Th5, ThOut, m_trn, n_trn):
    X_trn = X_trn.col_insert(0, ones(m_trn, 1))
    a2 = X_trn*Th1
    a2 = a2.applyfunc(sig)

    a2 = a2.col_insert(0, ones(m_trn, 1))
    a3 = a2*Th2
    a3 = a3.applyfunc(sig)

    a3 = a3.col_insert(0, ones(m_trn, 1))
    a4 = a3*Th3
    a4 = a4.applyfunc(sig)

    a4 = a4.col_insert(0, ones(m_trn, 1))
    a5 = a4*Th4
    a5 = a5.applyfunc(sig)

    a5 = a5.col_insert(0, ones(m_trn, 1))
    a6 = a5*Th5
    a6 = a6.applyfunc(sig)

    a6 = a6.col_insert(0, ones(m_trn, 1))
    h0 = a6*ThOut
    h0 = h0.applyfunc(sig)

    j1h0 = h0.applyfunc(mlog)
    j1h0 = j1h0.transpose()
    j2h0 = h0.applyfunc(sub1)
    j2h0 = j2h0.applyfunc(mlog)
    j2h0 = j2h0.transpose()
    j2y = y_trn.applyfunc(sub1)
    Jpart1 = y_trn*j1h0
    Jpart2 = j2y*j2h0
    J = Jpart1 + Jpart2
    J = np.array(J).astype(np.float64)
    J = np.sum(J)
    J = -(J/m_trn)

    # Regularisation
    Th1r = np.array(Th1).astype(np.float64)
    Th2r = np.array(Th2).astype(np.float64)
    Th3r = np.array(Th3).astype(np.float64)
    Th4r = np.array(Th4).astype(np.float64)
    Th5r = np.array(Th5).astype(np.float64)
    ThOutr = np.array(ThOut).astype(np.float64)
    Th1r = np.delete(Th1r, 0, 1)
    Th2r = np.delete(Th2r, 0, 1)
    Th3r = np.delete(Th3r, 0, 1)
    Th4r = np.delete(Th4r, 0, 1)
    Th5r = np.delete(Th5r, 0, 1)
    ThOutr = np.delete(ThOutr, 0, 1)
    Th1r = np.power(Th1r, 2)
    Th2r = np.power(Th2r, 2)
    Th3r = np.power(Th3r, 2)
    Th4r = np.power(Th4r, 2)
    Th5r = np.power(Th5r, 2)
    ThOutr = np.power(ThOutr, 2)

    ThetaReg = (np.sum(Th1r) + np.sum(Th2r) + np.sum(Th3r) + np.sum(Th4r) + np.sum(Th5r) + np.sum(ThOutr))
    J = J + ThetaReg
    print(J)
    return a2, a3, a4, a5, a6, h0

def gradient_descent(a2, a3, a4, a5, a6, h0, X_trn, y_trn, Th1, Th2, Th3, Th4, Th5, ThOut, m_trn, n_trn):
    d7 = h0 - y_trn
    d6 = (ThOut)*(d7.T)
    d6 = d6.T
    print(d6.shape)
    a61 = a6.applyfunc(sub1)
    print(a6.shape)
    print(a61.shape)
    a6dif = (a6*a61)
