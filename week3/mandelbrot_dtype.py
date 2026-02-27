import numpy as np
import time, statistics
from numba import njit


@njit
def mandelbrot_kernel_f32(x, y, max_iter):
    h = y.size
    w = x.size
    out = np.empty((h, w), dtype=np.int32)

    for i in range(h):
        ci = y[i]
        for j in range(w):
            cr = x[j]

            zr = np.float32(0.0)
            zi = np.float32(0.0)
            n = 0

            while n < max_iter and (zr*zr + zi*zi) <= np.float32(4.0):
                zr_new = zr*zr - zi*zi + cr
                zi = np.float32(2.0)*zr*zi + ci
                zr = zr_new
                n += 1

            out[i, j] = n

    return out


@njit
def mandelbrot_kernel_f64(x, y, max_iter):
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


def quantize_to_float16_then_float32(arr_f32):
    """
    Simulate float16 precision on CPU:
    store/round as float16, then convert back to float32 for Numba kernel.
    """
    return arr_f32.astype(np.float16).astype(np.float32)


def bench(fn, *args, runs=10):
    fn(*args)  # warm-up (JIT compile)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
    width = height = 1024
    max_iter = 100

    # float32 (true)
    x32 = np.linspace(xmin, xmax, width).astype(np.float32)
    y32 = np.linspace(ymin, ymax, height).astype(np.float32)

    # float16 precision (quantized), computed in float32 kernel
    x16q = quantize_to_float16_then_float32(x32)
    y16q = quantize_to_float16_then_float32(y32)

    # float64 (true)
    x64 = np.linspace(xmin, xmax, width).astype(np.float64)
    y64 = np.linspace(ymin, ymax, height).astype(np.float64)

    t16 = bench(mandelbrot_kernel_f32, x16q, y16q, max_iter, runs=10)
    t32 = bench(mandelbrot_kernel_f32, x32, y32, max_iter, runs=10)
    t64 = bench(mandelbrot_kernel_f64, x64, y64, max_iter, runs=10)

    print(f"float16 precision (quantized, computed as float32): {t16:.6f} s")
    print(f"float32:                                          {t32:.6f} s")
    print(f"float64:                                          {t64:.6f} s")