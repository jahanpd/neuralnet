import numpy as np
from sympy import *
from sigmoid import sigmoid
import mpmath as mp

sig = lambda x: 1.0/(1.0+mp.exp(x))
mlog = lambda x: mp.log(x)
sub1 = lambda x: 1.0-x

def costFunction(X_trn, y_trn, Th1, Th2, Th3, Th4, Th5, ThOut, m_trn, n_trn):
    X_trn = Matrix(X_trn)
    print(X_trn.shape)
    X_trn = X_trn.col_insert(0, ones(m_trn, 1))
    Th1 = Matrix(Th1)
    a2 = X_trn*Th1    
    a2 = a2.applyfunc(sig)
    print(a2.shape)
    
    a2 = a2.col_insert(0, ones(m_trn, 1))
    Th2 = Matrix(Th2)
    a3 = a2*Th2
    a3 = a3.applyfunc(sig)
    print(a3.shape)

    a3 = a3.col_insert(0, ones(m_trn, 1))
    Th3 = Matrix(Th3)
    a4 = a3*Th3
    a4 = a4.applyfunc(sig)
    print(a4.shape)
    
    a4 = a4.col_insert(0, ones(m_trn, 1))
    Th4 = Matrix(Th4)
    a5 = a4*Th4
    a5 = a5.applyfunc(sig)
    print(a5.shape)
    
    a5 = a5.col_insert(0, ones(m_trn, 1))
    Th5 = Matrix(Th5)
    a6 = a5*Th5
    a6 = a6.applyfunc(sig)
    print(a6.shape)
    
    a6 = a6.col_insert(0, ones(m_trn, 1))
    ThOut = Matrix(ThOut)
    h0 = a6*ThOut
    h0 = h0.applyfunc(sig)
    print(h0.shape)
    
    
    y_trn = Matrix(y_trn)
    print(y_trn.shape)
    j1h0 = h0.applyfunc(mlog)
    j1h0 = j1h0.transpose()
    print(j1h0.shape)
    j2h0 = h0.applyfunc(sub1)
    j2h0 = j2h0.applyfunc(mlog)
    j2h0 = j2h0.transpose()
    j2y = y_trn.applyfunc(sub1)
    Jpart1 = y_trn*j1h0
    print(Jpart1.shape)
    Jpart2 = j2y*j2h0
    print(Jpart2.shape)
    J = Jpart1 + Jpart2
    J = np.array(J).astype(np.float64)
    J = np.sum(J)
    J = -(J/m_trn)
    print(J)
