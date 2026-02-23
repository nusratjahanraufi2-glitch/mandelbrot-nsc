"""
Milestone 4: Problem Size Scaling (Week 2)

Task (from slides):
- Run vectorized Mandelbrot for grid sizes: 256, 512, 1024, 2048, 4096
- Record runtime for each (use median of >=3 runs)
- Plot grid size vs runtime
- Predict: if 1024x1024 takes X, should 2048x2048 take ~4X? (4x pixels)
"""

import statistics
import time
import matplotlib.pyplot as plt

from mandelbrot_vectorized import mandelbrot_vectorized


def benchmark(func, *args, n_runs=3, **kwargs):
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    print(f"Median over {n_runs} runs: {med:.4f}s (min={min(times):.4f}, max={max(times):.4f})")
    return med, result


def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    max_iter = 200

    sizes = [256, 512, 1024, 2048, 4096]

    grid_sizes = []
    runtimes = []

    print("Running vectorized Mandelbrot scaling test...\n")

    for n in sizes:
        print(f"Grid size: {n} x {n}")
        t_med, _ = benchmark(
            mandelbrot_vectorized,
            xmin, xmax, ymin, ymax, n, n,
            max_iter=max_iter,
            n_runs=3
        )
        grid_sizes.append(n)
        runtimes.append(t_med)
        print()

    # Print simple prediction check: 2048 should be ~4x 1024 (if perfect scaling)
    if 1024 in grid_sizes and 2048 in grid_sizes:
        t1024 = runtimes[grid_sizes.index(1024)]
        t2048 = runtimes[grid_sizes.index(2048)]
        print("Prediction check:")
        print(f"1024x1024 median: {t1024:.4f}s")
        print(f"2048x2048 median: {t2048:.4f}s")
        print(f"Expected ~4x if purely pixel-scaling: {4*t1024:.4f}s")
        print(f"Observed ratio (2048/1024): {t2048/t1024:.2f}x")
        print()

    # Plot
    plt.figure()
    plt.plot(grid_sizes, runtimes, marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Grid size N (N x N)")
    plt.ylabel("Median runtime (seconds)")
    plt.title("Mandelbrot scaling (vectorized)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()