import numpy as np
from sympy import *
import mpmath as mp

sub1 = lambda x: 1.0-x
sig = lambda x: 1.0/(1.0+mp.exp(x))

def sigdif(z):
    sigmoid = z.applyfunc(sig)
    zsub1 = sigmoid.applyfunc(sub1)
    out = sigmoid.multiply_elementwise(zsub1)
    return out
