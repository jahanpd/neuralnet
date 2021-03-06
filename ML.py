import data as d
import initialTheta
import numpy as np
from sympy import *
import costFunction as cf
import os

__file__ = 'ML/BreastMass/bcfna.csv'
csv = os.path.realpath(
    os.path.join(os.getcwd(), __file__))

X_trn, X_tst, y_trn, y_tst, m_trn, n_trn, m_tst, n_tst = d.import_data(csv)

y_trn = d.transform_y(y_trn)
y_trn = Matrix(y_trn)

X_trn = d.normalise(X_trn)
X_trn = Matrix(X_trn)

# hidden layers
hl = 5
# nodes in each hidden layer
nl = 10
# number of outputs
out = 2

# alpha for gradient descent
alpha = lambda x: 0.0001*x

# lambda Regularisation
lmbd = 1

# generate thetas, initialise to a random value between [-eps,eps]
Th1 = initialTheta.Theta1(n_trn, nl, out)

hl_thetas = initialTheta.ThetaHL(n_trn, nl, out, hl)
Th2, Th3, Th4, Th5 = np.array_split(hl_thetas, 4)
Th1 = Matrix(Th1)
Th2 = Matrix(np.reshape(Th2, (nl+1, nl)))
Th3 = Matrix(np.reshape(Th3, (nl+1, nl)))
Th4 = Matrix(np.reshape(Th4, (nl+1, nl)))
Th5 = Matrix(np.reshape(Th5, (nl+1, nl)))

ThOut = initialTheta.ThetaOut(n_trn, nl, out)
ThOut = Matrix(ThOut)

count = 0
while count < 100:
    J = cf.costFunction(X_trn, y_trn, Th1, Th2, Th3, Th4, Th5, ThOut, m_trn,
        n_trn, lmbd)

    Th1Grad, Th2Grad, Th3Grad, Th4Grad, Th5Grad, ThOutGrad = \
        cf.gradient_descent(X_trn, y_trn, Th1, Th2, Th3, Th4, Th5, ThOut,
            m_trn, n_trn)

    print(J)
    Th1 = Th1 - (Th1Grad.applyfunc(alpha))
    Th2 = Th2 - (Th2Grad.applyfunc(alpha))
    Th3 = Th3 - (Th3Grad.applyfunc(alpha))
    Th4 = Th4 - (Th4Grad.applyfunc(alpha))
    Th5 = Th5 - (Th5Grad.applyfunc(alpha))
    ThOut = ThOut - (ThOutGrad.applyfunc(alpha))
    count += 1
