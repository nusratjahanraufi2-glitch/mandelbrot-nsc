import numpy as np
import os
from multiprocessing import Pool
from numba import njit


@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_iter):
        z_real2 = z_real * z_real - z_imag * z_imag + c_real
        z_imag2 = 2.0 * z_real * z_imag + c_imag

        z_real = z_real2
        z_imag = z_imag2

        if z_real * z_real + z_imag * z_imag > 4.0:
            return i

    return max_iter


@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    result = np.empty((row_end - row_start, N), dtype=np.uint16)

    for local_row, row in enumerate(range(row_start, row_end)):
        c_imag = y_min + (y_max - y_min) * row / (N - 1)

        for col in range(N):
            c_real = x_min + (x_max - x_min) * col / (N - 1)
            result[local_row, col] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return result


def _worker(args):
    return mandelbrot_chunk(*args)


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def mandelbrot_parallel(
    N, x_min, x_max, y_min, y_max,
    max_iter=100, n_workers=4, n_chunks=None, pool=None
):
    if n_chunks is None:
        n_chunks = n_workers

    chunk_size = max(1, N // n_chunks)

    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    if pool is not None:
        return np.vstack(pool.map(_worker, chunks))

    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]

    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny)
        parts = p.map(_worker, chunks)

    return np.vstack(parts)


if __name__ == "__main__":
    N = 1024
    max_iter = 100
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    n_workers = max(1, os.cpu_count() // 2)
    n_chunks = 4 * n_workers

    serial_result = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    parallel_result = mandelbrot_parallel(
        N,
        X_MIN,
        X_MAX,
        Y_MIN,
        Y_MAX,
        max_iter=max_iter,
        n_workers=n_workers,
        n_chunks=n_chunks
    )

    print("Serial shape:", serial_result.shape)
    print("Parallel shape:", parallel_result.shape)
    print("Same result:", np.array_equal(serial_result, parallel_result))
    print("n_workers:", n_workers)
    print("n_chunks:", n_chunks)