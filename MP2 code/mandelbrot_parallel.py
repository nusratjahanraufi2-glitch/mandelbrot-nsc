import numpy as np
from numba import njit


@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_iter):
        if z_real * z_real + z_imag * z_imag > 4.0:
            return i

        old_real = z_real
        old_imag = z_imag

        z_real = old_real * old_real - old_imag * old_imag + c_real
        z_imag = 2.0 * old_real * old_imag + c_imag

    return max_iter


@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)

    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy

        for col in range(N):
            c_real = x_min + col * dx
            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def main():
    N = 1024
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25
    MAX_ITER = 100

    image = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)

    print("Shape:", image.shape)
    print("dtype:", image.dtype)
    print("min:", image.min())
    print("max:", image.max())


main()