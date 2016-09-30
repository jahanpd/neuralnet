import numpy as np
from sympy import *

sub1 = lambda x: 1.0-x


def sigdif(z):
    zsub1 = z.applyfunc(sub1)
    out = z.multiply_elementwise(zsub1)
    return out
