"""
Milestone 2: Timing comparison (Naive vs Vectorized Mandelbrot)
"""

import time
from mandelbrot_grid import compute_mandelbrot
from mandelbrot_vectorized import mandelbrot_vectorized


def time_run(label, func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"{label}: {elapsed:.3f} seconds")
    return elapsed


def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    max_iter = 200

    nx, ny = 1000, 1000

    print(f"Grid size: {nx} x {ny}")

    t_naive = time_run(
        "Naive",
        compute_mandelbrot,
        xmin, xmax, ymin, ymax,
        nx, ny,
        max_iter=max_iter
    )

    t_vec = time_run(
        "Vectorized",
        mandelbrot_vectorized,
        xmin, xmax, ymin, ymax,
        nx, ny,
        max_iter=max_iter
    )

    print(f"Speedup: {t_naive / t_vec:.2f}x faster")


if __name__ == "__main__":
    main()