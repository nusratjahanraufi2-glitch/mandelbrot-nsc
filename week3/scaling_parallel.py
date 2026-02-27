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
    max_iter = 300

    for N in [512, 1024, 2048]:
        x = np.linspace(xmin, xmax, N).astype(np.float64)
        y = np.linspace(ymin, ymax, N).astype(np.float64)

        t_serial = bench(mandelbrot_kernel_f64, x, y, max_iter, runs=5)
        t_parallel = bench(mandelbrot_parallel, x, y, max_iter, runs=5)

        print(f"N={N:4d} | serial={t_serial:.3f}s | parallel={t_parallel:.3f}s | speedup={t_serial/t_parallel:.2f}x")