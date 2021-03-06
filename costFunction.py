import numpy as np
from sympy import *
import mpmath as mp
from sigmoid import sigdif

# sympy elementwise functions

sig = lambda x: 1.0/(1.0+mp.exp(x))
mlog = lambda x: mp.log(x)
sub1 = lambda x: 1.0-x
sqar = lambda x: mp.power(x, 2)


def costFunction(X_trn, y_trn, Th1, Th2, Th3, Th4, Th5, ThOut, m_trn, n_trn,
    lmbd):
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
    j2h0 = h0.applyfunc(sub1)
    j2h0 = j2h0.applyfunc(mlog)
    j2y = y_trn.applyfunc(sub1)
    Jpart1 = y_trn.multiply_elementwise(j1h0)
    Jpart2 = j2y.multiply_elementwise(j2h0)
    J = Jpart1 - Jpart2
    J = np.array(J).astype(np.float64)
    J = np.sum(J, axis=1)
    J = (1/m_trn)*(np.sum(J))

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

    ThetaReg = (lmbd*(np.sum(Th1r) + np.sum(Th2r) + np.sum(Th3r) + np.sum(Th4r) +
    np.sum(Th5r) + np.sum(ThOutr)))/(2*m_trn)
    J = J + ThetaReg
    return J


def gradient_descent(X_trn, y_trn, Th1, Th2, Th3, Th4, Th5, ThOut, m_trn,
    n_trn):
    X_trn = X_trn.col_insert(0, ones(m_trn, 1))
    r, c = Th1.shape
    Th1Grad = zeros(r, c)
    r, c = Th2.shape
    Th2Grad = zeros(r, c)
    r, c = Th3.shape
    Th3Grad = zeros(r, c)
    r, c = Th4.shape
    Th4Grad = zeros(r, c)
    r, c = Th5.shape
    Th5Grad = zeros(r, c)
    r, c = ThOut.shape
    ThOutGrad = zeros(r, c)

    for t in range(m_trn):
        a1 = X_trn[t, :]
        z2 = a1*Th1
        a2 = z2.applyfunc(sig)
        r, c = a2.shape
        a2 = a2.col_insert(0, ones(r, 1))

        z3 = a2*Th2
        a3 = z3.applyfunc(sig)
        r, c = a3.shape
        a3 = a3.col_insert(0, ones(r, 1))

        z4 = a3*Th3
        a4 = z4.applyfunc(sig)
        r, c = a4.shape
        a4 = a4.col_insert(0, ones(r, 1))

        z5 = a4*Th4
        a5 = z5.applyfunc(sig)
        r, c = a5.shape
        a5 = a5.col_insert(0, ones(r, 1))

        z6 = a5*Th5
        a6 = z6.applyfunc(sig)
        r, c = a6.shape
        a6 = a6.col_insert(0, ones(r, 1))

        z7 = a6*ThOut
        a7 = z7.applyfunc(sig)

        d7 = a7-y_trn[t, :]

        d6 = (ThOut*(d7.T))
        r, c = z6.shape
        z6 = z6.col_insert(0, ones(r, 1))
        z6dif = sigdif(z6)
        d6 = d6.multiply_elementwise(z6dif.T)

        d6bp = d6
        d6bp.row_del(0)
        d5 = Th5*d6bp
        r, c = z5.shape
        z5 = z5.col_insert(0, ones(r, 1))
        z5dif = sigdif(z5)
        d5 = d5.multiply_elementwise(z5dif.T)

        d5bp = d5
        d5bp.row_del(0)
        d4 = (Th4*d5bp)
        r, c = z4.shape
        z4 = z4.col_insert(0, ones(r, 1))
        z4dif = sigdif(z4)
        d4 = d4.multiply_elementwise(z4dif.T)

        d4bp = d4
        d4bp.row_del(0)
        d3 = (Th3*d4bp)
        r, c = z3.shape
        z3 = z3.col_insert(0, ones(r, 1))
        z3dif = sigdif(z3)
        d3 = d3.multiply_elementwise(z3dif.T)

        d3bp = d3
        d3bp.row_del(0)
        d2 = (Th2*d3bp)
        r, c = z2.shape
        z2 = z2.col_insert(0, ones(r, 1))
        z2dif = sigdif(z2)
        d2 = d2.multiply_elementwise(z3dif.T)
        d2.row_del(0)

        d2 = d2.T
        a1 = a1.T
        r, c = a1.shape
        for x in range(r):
            Th1Grad[x, :] += a1[x, :]*d2

        d3 = d3.T
        a2 = a2.T
        r, c = a2.shape
        for x in range(r):
            Th2Grad[x, :] += a2[x, :]*d3

        d4 = d4.T
        a3 = a3.T
        r, c = a3.shape
        for x in range(r):
            Th3Grad[x, :] += a3[x, :]*d4

        d5 = d5.T
        a4 = a4.T
        r, c = a4.shape
        for x in range(r):
            Th4Grad[x, :] += a4[x, :]*d5

        d6 = d6.T
        a5 = a5.T
        r, c = a5.shape
        for x in range(r):
            Th5Grad[x, :] += a5[x, :]*d6

        a6 = a6.T
        r, c = a6.shape
        for x in range(r):
            ThOutGrad[x, :] += a6[x, :]*d7

        return Th1Grad, Th2Grad, Th3Grad, Th4Grad, Th5Grad, ThOutGrad
