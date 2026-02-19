import numpy as np
import time
import matplotlib.pyplot as plt


def mandelbrot_point(c: complex, max_iter: int = 200) -> int:
    z = 0 + 0j
    for i in range(max_iter):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) > 4.0:
            return i
    return max_iter


def compute_mandelbrot(xmin, xmax, ymin, ymax, nx, ny, max_iter=200):
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    result = np.empty((ny, nx), dtype=np.int32)

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            result[j, i] = mandelbrot_point(complex(x, y), max_iter=max_iter)

    return result


def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5

    nx = ny = 1000
    max_iter = 200

    start = time.perf_counter()
    img = compute_mandelbrot(xmin, xmax, ymin, ymax, nx, ny, max_iter)
    elapsed = time.perf_counter() - start

    print("Grid shape:", img.shape)
    print(f"Computation took {elapsed:.3f} seconds")

    # Plot (popup)
    plt.figure()
    plt.imshow(
        img,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",          # important: prevents upside-down image
        cmap="viridis"
    )
    plt.colorbar(label="Iteration count")
    plt.title("Mandelbrot set (naive)")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.tight_layout()
    plt.show()   # popup window


if __name__ == "__main__":
    main()
