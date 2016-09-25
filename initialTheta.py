import numpy as np
import math as mt


def Theta1(n_trn, nl, out):
    epsilon = mt.sqrt(6)/mt.sqrt(out + n_trn)
    Theta1 = np.random.randn(n_trn+1, nl)*(2*epsilon)-epsilon
    return Theta1


def ThetaHL(n_trn, nl, out, hl):
    epsilon = mt.sqrt(6)/mt.sqrt(out + n_trn)
    hl_thetas = hl-1
    nl_thetas = nl + 1
    thetas = np.random.randn(nl_thetas*nl*hl_thetas)*(2*epsilon)-epsilon
    return thetas


def ThetaOut(n_trn, nl, out):
    epsilon = mt.sqrt(6)/mt.sqrt(out + n_trn)
    nl_thetas = nl + 1
    ThetaOut = np.random.randn(nl_thetas, out)*(2*epsilon)-epsilon
    return ThetaOut
