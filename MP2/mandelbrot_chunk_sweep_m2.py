import time
import statistics
import os
from multiprocessing import Pool

from mandelbrot_chunked_parallel import (
    mandelbrot_serial,
    mandelbrot_parallel,
    mandelbrot_chunk,
    _worker
)


if __name__ == "__main__":

    N = 1024
    max_iter = 100
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    n_workers = max(1, os.cpu_count() // 2)

    print("Warming up JIT...")
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # Serial baseline
    serial_times = []

    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        serial_times.append(time.perf_counter() - t0)

    t_serial = statistics.median(serial_times)

    print(f"\nSerial median time: {t_serial:.4f} s")

    print("\nChunk sweep:")
    print(f"{'n_chunks':>10} | {'time (s)':>10} | {'speedup':>10} | {'LIF':>10}")

    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]

    for mult in [1, 2, 4, 8, 16]:

        n_chunks = mult * n_workers

        with Pool(processes=n_workers) as pool:

            pool.map(_worker, tiny)

            times = []

            for _ in range(3):

                t0 = time.perf_counter()

                mandelbrot_parallel(
                    N,
                    X_MIN,
                    X_MAX,
                    Y_MIN,
                    Y_MAX,
                    max_iter=max_iter,
                    n_workers=n_workers,
                    n_chunks=n_chunks,
                    pool=pool
                )

                times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)

        speedup = t_serial / t_par
        lif = n_workers * t_par / t_serial - 1

        print(f"{n_chunks:10d} | {t_par:10.4f} | {speedup:10.2f} | {lif:10.2f}")