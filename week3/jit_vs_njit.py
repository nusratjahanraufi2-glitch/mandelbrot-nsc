import time
import numpy as np
from numba import jit, njit

# settings (use same as your other benchmarks)
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
N = 2048          # you can try 1024 first if you want faster runs
max_iter = 100
runs = 5

x = np.linspace(xmin, xmax, N)
y = np.linspace(ymin, ymax, N)

# @jit allows object-mode fallback if nopython compilation fails
@jit
def mandelbrot_jit(x, y, max_iter):
    h = y.size
    w = x.size
    out = np.empty((h, w), dtype=np.int32)

    for i in range(h):
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

# @njit forces nopython mode
@njit
def mandelbrot_njit(x, y, max_iter):
    h = y.size
    w = x.size
    out = np.empty((h, w), dtype=np.int32)

    for i in range(h):
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


def bench(fn, name):
    # warm-up (compile)
    fn(x, y, max_iter)

    # timed runs (exclude compile)
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(x, y, max_iter)
    t1 = time.perf_counter()

    avg = (t1 - t0) / runs
    print(f"{name}: {avg:.6f} s (avg over {runs} runs)")
    return avg


if __name__ == "__main__":
    t_jit = bench(mandelbrot_jit, "@jit")
    t_njit = bench(mandelbrot_njit, "@njit")

    print(f"\nSpeed ratio (jit/njit): {t_jit / t_njit:.2f}x")

    # Useful diagnostic: whether @jit ended up in nopython mode anyway
    try:
        print("jit nopython signatures:", mandelbrot_jit.nopython_signatures)
    except Exception as e:
        print("Could not read jit nopython signatures:", e)