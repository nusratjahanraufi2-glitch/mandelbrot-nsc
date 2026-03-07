from mandelbrot_week3_m2 import mandelbrot_naive

if __name__ == "__main__":
    M = mandelbrot_naive(-2, 1, -1.5, 1.5, 512, 512, 100)
    print(M.shape, M.dtype)