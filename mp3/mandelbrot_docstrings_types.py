import numpy as np


def mandelbrot_pixel(c: complex, max_iter: int) -> int:
    """Compute the escape iteration count for one complex point.

    Iterates z_(n+1) = z_n^2 + c starting from z_0 = 0 until
    the magnitude of z exceeds 2 or the maximum number of
    iterations is reached.

    Parameters
    ----------
    c : complex
        Complex coordinate to test for Mandelbrot set membership.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    int
        Iteration count at which the orbit escapes, or max_iter
        if it does not escape.
    """
    z = 0j
    for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return max_iter


def mandelbrot_grid(
    n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int,
) -> np.ndarray:
    """Compute a 2D grid of Mandelbrot escape iteration counts.

    Parameters
    ----------
    n : int
        Number of grid points in each dimension.
    x_min : float
        Minimum value on the real axis.
    x_max : float
        Maximum value on the real axis.
    y_min : float
        Minimum value on the imaginary axis.
    y_max : float
        Maximum value on the imaginary axis.
    max_iter : int
        Maximum number of iterations per point.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n, n) containing iteration counts.
    """
    result = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        y = y_min + (y_max - y_min) * i / n
        for j in range(n):
            x = x_min + (x_max - x_min) * j / n
            c = x + 1j * y
            result[i, j] = mandelbrot_pixel(c, max_iter)

    return result