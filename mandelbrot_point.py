def mandelbrot_point(c: complex, max_iter: int = 200) -> int:
    """
    Returns the iteration count before escape for a single complex point c.
    If it never escapes, returns max_iter.
    """
    z = 0 + 0j
    for i in range(max_iter):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) > 4.0:  # |z| > 2  (using squared magnitude)
            return i
    return max_iter


def main():
    # Quick checks (these are common checks from lectures)
    print("c = 0:", mandelbrot_point(0+0j))               # should be max_iter (inside set)
    print("c = 2:", mandelbrot_point(2+0j))               # should escape quickly
    print("c = -1:", mandelbrot_point(-1+0j))             # inside set (usually max_iter)


if __name__ == "__main__":
    main()
