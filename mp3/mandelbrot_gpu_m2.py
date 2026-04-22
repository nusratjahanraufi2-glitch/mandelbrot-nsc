#!/usr/bin/env python3
"""
mandelbrot_gpu.py

MP3 M2 — Mandelbrot float32 vs float64 in PyOpenCL.
Runs both GPU kernels, times them, and saves both images.
"""

import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt


KERNEL_SRC_F32 = """
__kernel void mandelbrot_f32(
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


KERNEL_SRC_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0;
    double zi = 0.0;
    int count = 0;

    while (count < max_iter && zr * zr + zi * zi <= 4.0) {
        double tmp = zr * zr - zi * zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""


def run_mandelbrot_f32(ctx, queue, prog, n, max_iter, x_min, x_max, y_min, y_max):
    image = np.zeros((n, n), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)
    kernel = prog.mandelbrot_f32

    kernel(
        queue, (64, 64), None,
        image_dev,
        np.float32(x_min), np.float32(x_max),
        np.float32(y_min), np.float32(y_max),
        np.int32(64), np.int32(max_iter),
    )
    queue.finish()

    t0 = time.perf_counter()
    kernel(
        queue, (n, n), None,
        image_dev,
        np.float32(x_min), np.float32(x_max),
        np.float32(y_min), np.float32(y_max),
        np.int32(n), np.int32(max_iter),
    )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()
    return image, elapsed


def run_mandelbrot_f64(ctx, queue, prog, n, max_iter, x_min, x_max, y_min, y_max):
    image = np.zeros((n, n), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)
    kernel = prog.mandelbrot_f64

    kernel(
        queue, (64, 64), None,
        image_dev,
        np.float64(x_min), np.float64(x_max),
        np.float64(y_min), np.float64(y_max),
        np.int32(64), np.int32(max_iter),
    )
    queue.finish()

    t0 = time.perf_counter()
    kernel(
        queue, (n, n), None,
        image_dev,
        np.float64(x_min), np.float64(x_max),
        np.float64(y_min), np.float64(y_max),
        np.int32(n), np.int32(max_iter),
    )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()
    return image, elapsed


def main():
    MAX_ITER = 200
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    dev = ctx.devices[0]

    print(f"Device: {dev.name}")
    if "cl_khr_fp64" not in dev.extensions:
        print("No native fp64 support detected; expect large slowdown or limited support.")

    prog_f32 = cl.Program(ctx, KERNEL_SRC_F32).build()
    prog_f64 = cl.Program(ctx, KERNEL_SRC_F64).build()

    for n in (1024, 2048):
        print(f"\n--- N = {n} ---")

        image_f32, t_f32 = run_mandelbrot_f32(
            ctx, queue, prog_f32, n, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX
        )
        print(f"GPU f32 {n}x{n}: {t_f32 * 1e3:.1f} ms")

        image_f64, t_f64 = run_mandelbrot_f64(
            ctx, queue, prog_f64, n, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX
        )
        print(f"GPU f64 {n}x{n}: {t_f64 * 1e3:.1f} ms")

        if t_f32 > 0:
            print(f"f64 / f32 speed ratio: {t_f64 / t_f32:.2f}")

        plt.imshow(image_f32, cmap="hot", origin="lower")
        plt.axis("off")
        plt.savefig(f"mandelbrot_gpu_f32_{n}.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.imshow(image_f64, cmap="hot", origin="lower")
        plt.axis("off")
        plt.savefig(f"mandelbrot_gpu_f64_{n}.png", dpi=150, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()