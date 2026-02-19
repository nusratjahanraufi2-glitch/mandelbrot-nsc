import numpy as np
import time
import matplotlib.pyplot as plt


def mandelbrot_vectorized(xmin, xmax, ymin, ymax, nx, ny, max_iter=200):
    # Create grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # Initialize Z and iteration counter
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)

    # Only ONE loop (iterations)
    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1

    return M


def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5

    nx = ny = 1000
    max_iter = 200

    start = time.perf_counter()
    img = mandelbrot_vectorized(xmin, xmax, ymin, ymax, nx, ny, max_iter)
    elapsed = time.perf_counter() - start

    print(f"Vectorized computation took {elapsed:.3f} seconds")

    plt.imshow(
        img,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap="viridis"
    )
    plt.colorbar(label="Iteration count")
    plt.title("Mandelbrot (Vectorized)")
    plt.show()


if __name__ == "__main__":
    main()
