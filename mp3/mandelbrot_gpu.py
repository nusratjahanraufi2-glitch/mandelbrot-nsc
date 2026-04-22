#!/usr/bin/env python3
"""
mandelbrot_gpu.py

MP3 M1 — Mandelbrot float32 kernel in PyOpenCL.
Builds and runs a GPU Mandelbrot kernel, times it, and saves the image.
"""

import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt


KERNEL_SRC = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f;
    float zi = 0.0f;
    int count = 0;

    while (count < max_iter && zr * zr + zi * zi <= 4.0f) {
        float tmp = zr * zr - zi * zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""


def main():
    N = 1024
    MAX_ITER = 200
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    print(f"Device: {ctx.devices[0].name}")

    prog = cl.Program(ctx, KERNEL_SRC).build()
    mandelbrot_kernel = prog.mandelbrot

    image = np.zeros((N, N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    # --- Warm up (first launch triggers a kernel compile) ---
    mandelbrot_kernel(
        queue, (64, 64), None, image_dev,
        np.float32(X_MIN), np.float32(X_MAX),
        np.float32(Y_MIN), np.float32(Y_MAX),
        np.int32(64), np.int32(MAX_ITER),
    )
    queue.finish()

    # --- Time the real run ---
    t0 = time.perf_counter()
    mandelbrot_kernel(
        queue, (N, N), None, image_dev,
        np.float32(X_MIN), np.float32(X_MAX),
        np.float32(Y_MIN), np.float32(Y_MAX),
        np.int32(N), np.int32(MAX_ITER),
    )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()

    print(f"GPU {N}x{N}: {elapsed * 1e3:.1f} ms")

    plt.imshow(image, cmap="hot", origin="lower")
    plt.axis("off")
    plt.savefig("mandelbrot_gpu.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()