import numpy as np
import time
import statistics
import logging
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
from numba import njit
from multiprocessing import Pool

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado").setLevel(logging.ERROR)


# -------------------------------------------------
# 1. Naive Python
# -------------------------------------------------
def mandelbrot_naive(N, x_min, x_max, y_min, y_max, max_iter=100):
    result = np.zeros((N, N), dtype=np.int32)

    for i in range(N):
        y = y_min + (y_max - y_min) * i / N
        for j in range(N):
            x = x_min + (x_max - x_min) * j / N
            c = complex(x, y)

            z = 0j
            count = 0
            while abs(z) <= 2 and count < max_iter:
                z = z * z + c
                count += 1

            result[i, j] = count

    return result


# -------------------------------------------------
# 2. NumPy
# -------------------------------------------------
def mandelbrot_numpy(N, x_min, x_max, y_min, y_max, max_iter=100):
    x = x_min + (x_max - x_min) * np.arange(N) / N
    y = y_min + (y_max - y_min) * np.arange(N) / N
    C = x + y[:, None] * 1j

    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] * Z[mask] + C[mask]

        escaped = (np.abs(Z) > 2) & (M == 0)
        M[escaped] = i + 1

    M[M == 0] = max_iter
    return M
# -------------------------------------------------
# 3. Numba
# -------------------------------------------------
@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    result = np.zeros((row_end - row_start, N), dtype=np.int32)

    for i in range(row_start, row_end):
        y = y_min + (y_max - y_min) * i / N

        for j in range(N):
            x = x_min + (x_max - x_min) * j / N
            c = complex(x, y)

            z = 0.0 + 0.0j
            count = 0

            while (z.real * z.real + z.imag * z.imag) <= 4.0 and count < max_iter:
                z = z * z + c
                count += 1

            result[i - row_start, j] = count

    return result


def mandelbrot_numba(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


# -------------------------------------------------
# 4. Multiprocessing
# -------------------------------------------------
def _mp_worker(args):
    return mandelbrot_chunk(*args)


def mandelbrot_multiprocessing(
    N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=8, n_chunks=8
):
    chunk_size = max(1, N // n_chunks)
    chunk_args = []

    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunk_args.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        parts = pool.map(_mp_worker, chunk_args)

    return np.vstack(parts)


# -------------------------------------------------
# 5. Dask local (same approach as M1/M2)
# -------------------------------------------------
def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=8):
    chunk_size = max(1, N // n_chunks)

    tasks = []
    row = 0

    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(
            delayed(mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = row_end

    parts = dask.compute(*tasks)
    return np.vstack(parts)


# -------------------------------------------------
# Timing helper
# -------------------------------------------------
def median_time(func, runs=3):
    times = []
    result = None

    for _ in range(runs):
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)

    return statistics.median(times), result


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    N = 1024
    MAX_ITER = 100
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    N_WORKERS = 8
    BEST_N_CHUNKS = 8   # from M2

    # warm up Numba once
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)

    # Naive Python
    naive_time, ref = median_time(
        lambda: mandelbrot_naive(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    )

    # NumPy
    numpy_time, numpy_result = median_time(
        lambda: mandelbrot_numpy(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    )

    # Numba
    numba_time, numba_result = median_time(
        lambda: mandelbrot_numba(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    )

    # Multiprocessing
    mp_time, mp_result = median_time(
        lambda: mandelbrot_multiprocessing(
            N, X_MIN, X_MAX, Y_MIN, Y_MAX,
            max_iter=MAX_ITER,
            n_workers=N_WORKERS,
            n_chunks=BEST_N_CHUNKS
        )
    )

    # Dask local
    cluster = None
    client = None
    try:
        cluster = LocalCluster(
            n_workers=N_WORKERS,
            threads_per_worker=1,
            processes=True,
            silence_logs=logging.ERROR
        )
        client = Client(cluster)

        # worker warm-up, same as M1/M2
        client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

        dask_time, dask_result = median_time(
            lambda: mandelbrot_dask(
                N, X_MIN, X_MAX, Y_MIN, Y_MAX,
                max_iter=MAX_ITER,
                n_chunks=BEST_N_CHUNKS
            )
        )
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if cluster is not None:
            try:
                cluster.close()
            except Exception:
                pass

    # correctness
    print("Correctness checks:")
    print("NumPy:", np.array_equal(ref, numpy_result))
    print("Numba:", np.array_equal(ref, numba_result))
    print("Multiprocessing:", np.array_equal(ref, mp_result))
    print("Dask local:", np.array_equal(ref, dask_result))

    implementations = [
        ("Naive Python", naive_time),
        ("NumPy", numpy_time),
        ("Numba (@njit)", numba_time),
        ("Numba + multiprocessing", mp_time),
        ("Dask local", dask_time),
    ]

    print("\nFull Benchmark – All Implementations")
    print("Implementation              | Time (s) | Speedup vs naive")
    print("-" * 58)

    for name, t in implementations:
        speedup = naive_time / t
        print(f"{name:27s} | {t:8.3f} | {speedup:16.2f}")