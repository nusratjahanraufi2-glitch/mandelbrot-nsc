"""
Mandelbrot Week 3 - Profiling & Optimization
Author: Nusrat Jahan Raufi
Course: Numerical Scientific Computing 2026
"""

import numpy as np


def mandelbrot_point(c, max_iter=100):
    z = 0j
    for n in range(max_iter):
        if (z.real * z.real + z.imag * z.imag) > 4.0:
            return n
        z = z*z + c
    return max_iter


def mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point(c, max_iter)

    return result


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