import numpy as np
import time, statistics
from mandelbrot_dtype import mandelbrot_kernel_f64
from mandelbrot_parallel import mandelbrot_parallel


def bench(fn, *args, runs=5):
    fn(*args)  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
    width = height = 2048   # increase workload for parallel benefit
    max_iter = 300

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    t_serial = bench(mandelbrot_kernel_f64, x, y, max_iter)
    t_parallel = bench(mandelbrot_parallel, x, y, max_iter)

    print(f"Serial Numba:   {t_serial:.3f} s")
    print(f"Parallel Numba: {t_parallel:.3f} s")
    print(f"Speedup:        {t_serial/t_parallel:.2f}x")