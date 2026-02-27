import time
import statistics

from mandelbrot_week3_numba import (
    mandelbrot_naive_py,
    mandelbrot_numpy,
    mandelbrot_hybrid,
    mandelbrot_naive_numba,
)

def bench(fn, *args, runs=5):
    # warm-up run (important especially for Numba JIT)
    fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
    width, height, max_iter = 512, 512, 100
    args = (xmin, xmax, ymin, ymax, width, height, max_iter)

    # extra warm-up for JIT compilation (exclude from timing)
    mandelbrot_hybrid(-2, 1, -1.5, 1.5, 64, 64, 100)
    mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64, 100)

    t_naive = bench(mandelbrot_naive_py, *args)
    t_numpy = bench(mandelbrot_numpy, *args)
    t_hybrid = bench(mandelbrot_hybrid, *args)
    t_numba = bench(mandelbrot_naive_numba, *args)

    print(f"Naive Python:   {t_naive:.3f} s")
    print(f"NumPy:          {t_numpy:.3f} s   (speedup {t_naive/t_numpy:.1f}x)")
    print(f"Numba Hybrid:   {t_hybrid:.3f} s  (vs full {t_hybrid/t_numba:.1f}x slower)")
    print(f"Numba Full:     {t_numba:.3f} s  (speedup {t_naive/t_numba:.1f}x)")