import numpy as np

def quadratic_naive(a, b, c):
    t = type(a)  # np.float32 or np.float64
    disc = t(np.sqrt(b*b - t(4)*a*c))   # keep dtype controlled
    x1 = (-b + disc) / (t(2)*a)
    x2 = (-b - disc) / (t(2)*a)
    return x1, x2

def quadratic_stable(a, b, c):
    t = type(a)
    disc = t(np.sqrt(b*b - t(4)*a*c))

    if b > 0:
        x1 = (-b - disc) / (t(2)*a)   # sign chosen to avoid cancellation
    else:
        x1 = (-b + disc) / (t(2)*a)

    x2 = c / (a * x1)   # Vieta's formula
    return x1, x2

true_small = 1.0 / 10000.0001

for dtype in [np.float32, np.float64]:
    a = dtype(1.0)
    b = dtype(-10000.0001)
    c = dtype(1.0)

    x1_naive, x2_naive = quadratic_naive(a, b, c)
    x1_stable, x2_stable = quadratic_stable(a, b, c)

    err_naive = abs(float(x2_naive) - true_small) / true_small
    err_stable = abs(float(x2_stable) - true_small) / true_small

    print(f"{dtype.__name__}:")
    print(f"  Naive  : x1 = {float(x1_naive):.4f}, x2 = {float(x2_naive):.10f}")
    print(f"  Stable : x1 = {float(x1_stable):.4f}, x2 = {float(x2_stable):.10f}")
    print(f"  Relative error (naive)  : {err_naive:.2e}")
    print(f"  Relative error (stable) : {err_stable:.2e}")
    print()