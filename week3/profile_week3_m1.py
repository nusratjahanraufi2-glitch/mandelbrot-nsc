import cProfile
import pstats
from mandelbrot_week3 import mandelbrot_naive, mandelbrot_numpy

# Run cProfile and save results
cProfile.run(
    "mandelbrot_naive(-2, 1, -1.5, 1.5, 512, 512, 100)",
    "naive_profile.prof"
)

cProfile.run(
    "mandelbrot_numpy(-2, 1, -1.5, 1.5, 512, 512, 100)",
    "numpy_profile.prof"
)

# Print top 10 by cumulative time
for name in ("naive_profile.prof", "numpy_profile.prof"):
    print(f"\n===== {name} (top 10 cumulative) =====")
    stats = pstats.Stats(name)
    stats.sort_stats("cumulative")
    stats.print_stats(10)