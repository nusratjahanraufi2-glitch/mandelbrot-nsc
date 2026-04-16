import numpy as np
import pytest

from mandelbrot_core import mandelbrot_grid, mandelbrot_pixel


def test_origin():
    assert mandelbrot_pixel(0 + 0j, 100) == 100


def test_far_outside():
    assert mandelbrot_pixel(5.0 + 0j, 100) == 1


@pytest.mark.parametrize(
    "c, max_iter, expected",
    [
        (0 + 0j, 100, 100),
        (5.0 + 0j, 100, 1),
        (-2.5 + 0j, 100, 1),
    ],
)
def test_known_cases(c, max_iter, expected):
    assert mandelbrot_pixel(c, max_iter) == expected


def test_grid_shape_and_type():
    grid = mandelbrot_grid(32, -2.0, 1.0, -1.5, 1.5, 50)
    assert grid.shape == (32, 32)
    assert isinstance(grid, np.ndarray)