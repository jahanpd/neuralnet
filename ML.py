import data as d
import initialTheta
import numpy as np
import costFunction as cf
import os

__file__='bcfna.csv'
csv = os.path.realpath(
    os.path.join(os.getcwd(), __file__))

X_trn, X_tst, y_trn, y_tst, m_trn, n_trn, m_tst, n_tst = d.import_data(csv)

y_trn = d.transform_y(y_trn)

X_trn = d.normalise(X_trn)

# hidden layers
hl = 5
# nodes in each hidden layer
nl = 100
# number of outputs
out = 2

# generate thetas, initialise to a random value between [-eps,eps]
Th1 = initialTheta.Theta1(n_trn, nl, out)

hl_thetas = initialTheta.ThetaHL(n_trn, nl, out, hl)
Th2, Th3, Th4 = np.array_split(hl_thetas, 3)
Th2 = np.reshape(Th2, (nl+1, nl))
Th2 = np.reshape(Th2, (nl+1, nl))
Th3 = np.reshape(Th3, (nl+1, nl))
Th3 = np.reshape(Th3, (nl+1, nl))
Th4 = np.reshape(Th4, (nl+1, nl))

ThOut = initialTheta.ThetaOut(n_trn, nl, out)

cf.costFunction(X_trn, y_trn, Th1, Th2, Th3, Th4, ThOut, m_trn, n_trn)
