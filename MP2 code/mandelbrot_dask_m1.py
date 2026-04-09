import numpy as np
import time
import statistics
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
from numba import njit


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

            while (z.real*z.real + z.imag*z.imag) <= 4.0 and count < max_iter:
                z = z*z + c
                count += 1

            result[i - row_start, j] = count

    return result


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32):
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


if __name__ == "__main__":
    N = 1024
    MAX_ITER = 100

    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    N_WORKERS = 8
    N_CHUNKS = 32

    # Warm up Numba locally
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)

    # Serial reference
    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)

    cluster = None
    client = None

    try:
        cluster = LocalCluster(
            n_workers=N_WORKERS,
            threads_per_worker=1,
            processes=True,
            silence_logs=True
        )
        client = Client(cluster)

        # Warm up Numba on workers
        client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

        times = []

        for _ in range(3):
            t0 = time.perf_counter()

            result = mandelbrot_dask(
                N, X_MIN, X_MAX, Y_MIN, Y_MAX,
                max_iter=MAX_ITER,
                n_chunks=N_CHUNKS
            )

            times.append(time.perf_counter() - t0)

        print(f"Dask local (chunks={N_CHUNKS}): {statistics.median(times):.3f} s")
        print("Same result:", np.array_equal(ref, result))

    finally:
        if client is not None:
            try:
                client.shutdown()
            except:
                pass
            client.close()

        if cluster is not None:
            cluster.close()