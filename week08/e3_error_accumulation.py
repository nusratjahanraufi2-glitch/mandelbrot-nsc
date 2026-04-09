import numpy as np

n_values = [10, 100, 1_000, 10_000, 100_000]

for dtype in [np.float32, np.float64]:
    print(f"\n{dtype.__name__}:")
    for n in n_values:
        total = dtype(0.0)
        for _ in range(n):
            total += dtype(0.1)

        expected = n * 0.1
        rel_error = abs(float(total) - expected) / expected

        print(f" n={n:>7d}: result={float(total):.10f} rel_error={rel_error:.2e}")