import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
import logging
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from numba import njit

# Reduce logs
logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado").setLevel(logging.ERROR)


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


def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter, n_chunks):
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

    parts = compute(*tasks)
    return np.vstack(parts)


if __name__ == "__main__":
    N = 1024
    MAX_ITER = 100

    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    N_WORKERS = 8
    CHUNKS_LIST = [4, 8, 16, 32, 64, 128]

    # Warm up locally
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)

    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.ERROR
    )
    client = Client(cluster)

    # Warm up workers
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    # Baseline (1 chunk)
    t1_runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER, n_chunks=1)
        t1_runs.append(time.perf_counter() - t0)

    T1 = statistics.median(t1_runs)

    print("n_chunks | time(s) | speedup | LIF")

    results = []

    for n_chunks in CHUNKS_LIST:
        times = []

        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_dask(
                N, X_MIN, X_MAX, Y_MIN, Y_MAX,
                MAX_ITER, n_chunks
            )
            times.append(time.perf_counter() - t0)

        tp = statistics.median(times)
        speedup = T1 / tp
        lif = N_WORKERS * (tp / T1) - 1

        results.append((n_chunks, tp))

        print(f"{n_chunks:8d} | {tp:7.3f} | {speedup:7.2f} | {lif:6.2f}")

    # Find best
    best = min(results, key=lambda x: x[1])
    print("\nBest n_chunks:", best[0])
    print("Best time:", best[1])

    # Plot
    x = [r[0] for r in results]
    y = [r[1] for r in results]

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel("n_chunks")
    plt.ylabel("time (s)")
    plt.title("Chunk sweep")
    plt.grid()
    plt.savefig("dask_chunk_sweep.png")
    plt.close()

    client.close()
    cluster.close()