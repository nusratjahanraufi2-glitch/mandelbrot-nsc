from mandelbrot_week3 import mandelbrot_naive

if __name__ == "__main__":
    # Smaller grid because line profiler is extremely slow (per lecture slides)
    mandelbrot_naive(-2, 1, -1.5, 1.5, 256, 256, 100)