# Wrapper for simple DDM simulator

import ctypes
import numpy as np

# You may need to compile ../c/wdm.so with:
#   gcc -shared -o wdm.so wdm.c

wdm = ctypes.CDLL('c/wdm.so')
wdm.rnd.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
wdm.rnd.restype = ctypes.POINTER(ctypes.c_double)

def wdmrnd(a, v, t, n):
    y = np.ctypeslib.as_array(wdm.rnd(a, t, 0.5, v, n), shape=(n,))
    return abs(y), (y > 0)

