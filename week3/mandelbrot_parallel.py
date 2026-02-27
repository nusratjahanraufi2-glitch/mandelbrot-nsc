import numpy as np
from numba import njit, prange


@njit(parallel=True)
def mandelbrot_parallel(x, y, max_iter):
    h = y.size
    w = x.size
    out = np.empty((h, w), dtype=np.int32)

    for i in prange(h):   # <-- parallel loop
        ci = y[i]
        for j in range(w):
            cr = x[j]

            zr = 0.0
            zi = 0.0
            n = 0

            while n < max_iter and (zr*zr + zi*zi) <= 4.0:
                zr_new = zr*zr - zi*zi + cr
                zi = 2.0*zr*zi + ci
                zr = zr_new
                n += 1

            out[i, j] = n

    return out