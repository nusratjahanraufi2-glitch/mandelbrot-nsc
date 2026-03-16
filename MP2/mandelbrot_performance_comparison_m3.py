import time
import numpy as np
from numba import njit
import os

from mandelbrot_chunked_parallel import mandelbrot_parallel


# ------------------------------
# Naive Python implementation
# ------------------------------
def mandelbrot_naive(N, x_min, x_max, y_min, y_max, max_iter):

    result = [[0]*N for _ in range(N)]

    for row in range(N):
        c_imag = y_min + (y_max - y_min) * row / (N - 1)

        for col in range(N):

            c_real = x_min + (x_max - x_min) * col / (N - 1)

            z_real = 0
            z_imag = 0

            count = 0

            while z_real*z_real + z_imag*z_imag <= 4 and count < max_iter:

                temp = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2*z_real*z_imag + c_imag
                z_real = temp

                count += 1

            result[row][col] = count

    return result


# ------------------------------
# NumPy vectorized version
# ------------------------------
def mandelbrot_numpy(N, x_min, x_max, y_min, y_max, max_iter):

    real = np.linspace(x_min, x_max, N)
    imag = np.linspace(y_min, y_max, N)

    C = real + imag[:, None] * 1j
    Z = np.zeros_like(C)

    output = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):

        mask = np.abs(Z) <= 2

        Z[mask] = Z[mask]**2 + C[mask]
        output[mask] = i

    return output


# ------------------------------
# Numba implementation
# ------------------------------
@njit
def mandelbrot_numba(N, x_min, x_max, y_min, y_max, max_iter):

    result = np.zeros((N, N), dtype=np.int32)

    for row in range(N):

        c_imag = y_min + (y_max - y_min) * row / (N - 1)

        for col in range(N):

            c_real = x_min + (x_max - x_min) * col / (N - 1)

            z_real = 0
            z_imag = 0

            count = 0

            while z_real*z_real + z_imag*z_imag <= 4 and count < max_iter:

                temp = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2*z_real*z_imag + c_imag
                z_real = temp

                count += 1

            result[row, col] = count

    return result


# ------------------------------
# Benchmark helper
# ------------------------------
def benchmark(func, *args):

    t0 = time.perf_counter()
    func(*args)
    return time.perf_counter() - t0


# ------------------------------
# Main benchmark
# ------------------------------
if __name__ == "__main__":

    N = 1024
    max_iter = 100

    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    n_workers = max(1, os.cpu_count() // 2)
    n_chunks = 16 * n_workers

    print("Running performance comparison...\n")

    # Naive
    t_naive = benchmark(
        mandelbrot_naive,
        N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter
    )

    # NumPy
    t_numpy = benchmark(
        mandelbrot_numpy,
        N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter
    )

    # Numba
    mandelbrot_numba(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)  # warmup
    t_numba = benchmark(
        mandelbrot_numba,
        N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter
    )

    # Multiprocessing (best from M2)
    t_parallel = benchmark(
        mandelbrot_parallel,
        N, X_MIN, X_MAX, Y_MIN, Y_MAX,
        max_iter,
        n_workers,
        n_chunks
    )

    print("Implementation | Time (s)")
    print("-------------------------")
    print(f"Naive Python   | {t_naive:.4f}")
    print(f"NumPy          | {t_numpy:.4f}")
    print(f"Numba          | {t_numba:.4f}")
    print(f"Multiprocessing| {t_parallel:.4f}")