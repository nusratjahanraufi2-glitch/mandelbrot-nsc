"""
Mandelbrot Week 3 - Milestone 3 (Numba)
Author: Nusrat Jahan Raufi
Course: Numerical Scientific Computing 2026
"""

import numpy as np
from numba import njit


# ---------- Naive Python (baseline) ----------
def mandelbrot_point_py(c, max_iter=100):
    z = 0j
    for n in range(max_iter):
        if (z.real * z.real + z.imag * z.imag) > 4.0:
            return n
        z = z * z + c
    return max_iter


def mandelbrot_naive_py(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point_py(c, max_iter)

    return result


# ---------- NumPy vectorized ----------
def mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)

    for _ in range(max_iter):
        mask = (Z.real * Z.real + Z.imag * Z.imag) <= 4.0
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        M[mask] += 1

    return M


# ---------- Numba (Approach A: Hybrid) ----------
@njit
def mandelbrot_point_numba(c, max_iter=100):
    z = 0j
    for n in range(max_iter):
        if (z.real * z.real + z.imag * z.imag) > 4.0:
            return n
        z = z * z + c
    return max_iter


def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    # outer loops still Python -> slower than fully compiled
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point_numba(c, max_iter)

    return result


# ---------- Numba (Approach B: Fully compiled - recommended) ----------
@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n

    return result