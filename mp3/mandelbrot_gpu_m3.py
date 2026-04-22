#!/usr/bin/env python3
"""
MP3 M3 — Benchmark comparison (log-scale bar chart)
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Runtimes (seconds) ---
labels = [
    "Naive Python",
    "NumPy",
    "Numba",
    "Multiprocessing",
    "Dask local",
    "GPU f32",
    "GPU f64",
]

times = np.array([
    4.785,    # Naive
    0.575,    # NumPy
    0.019,    # Numba
    4.132,    # Multiprocessing
    0.254,    # Dask local
    0.0031,   # GPU f32
    0.0092,   # GPU f64
])

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.bar(labels, times)
plt.yscale("log")
plt.ylabel("Time (seconds, log scale)")
plt.title("Mandelbrot Performance Comparison (MP3 M3)")
plt.xticks(rotation=30)

# Annotate bars
for i, v in enumerate(times):
    plt.text(i, v, f"{v:.3g}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("mandelbrot_benchmark.png", dpi=150)
plt.show()