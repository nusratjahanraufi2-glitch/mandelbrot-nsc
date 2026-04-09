import numpy as np
import time
import logging
import warnings
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from numba import njit

# Clean output
logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# -----------------------------
# Serial Numba baseline
# -----------------------------
@njit
def mandelbrot_serial(N, max_iter):
    result = np.zeros((N, N), dtype=np.int32)

    for i in range(N):
        y = -1.5 + (3.0 * i / N)
        for j in range(N):
            x = -2.0 + (3.0 * j / N)

            c = complex(x, y)
            z = 0.0j

            for k in range(max_iter):
                z = z * z + c
                if (z.real * z.real + z.imag * z.imag) > 4:
                    result[i, j] = k
                    break
            else:
                result[i, j] = max_iter

    return result


# -----------------------------
# Dask chunk computation
# -----------------------------
@njit
def mandelbrot_point(x, y, max_iter):
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) > 4:
            return i
    return max_iter


@njit
def mandelbrot_chunk(x_vals, y_vals, max_iter):
    result = np.zeros((len(y_vals), len(x_vals)), dtype=np.int32)
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            result[i, j] = mandelbrot_point(x_vals[j], y_vals[i], max_iter)
    return result


def mandelbrot_dask(N, chunk_size, max_iter):
    x = np.linspace(-2, 1, N)
    y = np.linspace(-1.5, 1.5, N)

    tasks = []

    for i in range(0, N, chunk_size):
        for j in range(0, N, chunk_size):
            x_chunk = x[j:j + chunk_size]
            y_chunk = y[i:i + chunk_size]
            task = delayed(mandelbrot_chunk)(x_chunk, y_chunk, max_iter)
            tasks.append(task)

    compute(*tasks)


# -----------------------------
# Chunk Sweep
# -----------------------------
def chunk_sweep(N, max_iter):
    cluster = LocalCluster(threads_per_worker=1, dashboard_address=None)
    client = Client(cluster)

    chunk_sizes = [32, 64, 128, 256, 512]
    results = []

    print("\n--- Chunk Sweep ---")

    for c in chunk_sizes:
        times = []

        for _ in range(3):
            start = time.perf_counter()
            mandelbrot_dask(N, c, max_iter)
            times.append(time.perf_counter() - start)

        median_time = sorted(times)[1]
        results.append((c, median_time))
        print(f"Chunk {c}: {median_time:.2f} sec")

    client.close()
    return results


# -----------------------------
# Worker Scaling
# -----------------------------
def worker_scaling(N, chunk_size, max_iter):
    worker_counts = [1, 2, 4]
    results = []

    print("\n--- Worker Scaling ---")

    for w in worker_counts:
        cluster = LocalCluster(
            n_workers=w,
            threads_per_worker=1,
            dashboard_address=None
        )
        client = Client(cluster)

        times = []

        for _ in range(3):
            start = time.perf_counter()
            mandelbrot_dask(N, chunk_size, max_iter)
            times.append(time.perf_counter() - start)

        median_time = sorted(times)[1]
        results.append((w, median_time))
        print(f"Workers {w}: {median_time:.2f} sec")

        client.close()

    return results


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    N = 4096
    max_iter = 100

    print("\n=== Serial Numba Baseline ===")
    start = time.perf_counter()
    mandelbrot_serial(N, max_iter)
    serial_time = time.perf_counter() - start
    print(f"Serial time: {serial_time:.2f} sec")

    chunk_results = chunk_sweep(N, max_iter)
    best_chunk = min(chunk_results, key=lambda x: x[1])[0]

    print("\nBest chunk size:", best_chunk)

    worker_results = worker_scaling(N, best_chunk, max_iter)

    # Best Dask time
    best_dask_time = min(worker_results, key=lambda x: x[1])[1]

    print("\n=== Speedup ===")
    speedup = serial_time / best_dask_time
    print(f"Best Dask time: {best_dask_time:.2f} sec")
    print(f"Speedup vs serial: {speedup:.2f}x")