import time
import numpy as np
from mandelbrot_week3 import mandelbrot_naive, mandelbrot_numpy
from mandelbrot_numba_serial import mandelbrot_numba_serial

xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width = height = 512
max_iter = 100

x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)

def bench(fn, name):
    t0 = time.perf_counter()
    fn(x, y, max_iter) if name.startswith("Numba") else fn(xmin, xmax, ymin, ymax, width, height, max_iter)
    t1 = time.perf_counter()
    print(f"{name}: {t1 - t0:.3f} s")

# Warm-up Numba (compile)
mandelbrot_numba_serial(x, y, max_iter)

bench(mandelbrot_naive, "Naive")
bench(mandelbrot_numpy, "NumPy")
bench(mandelbrot_numba_serial, "Numba (serial)")