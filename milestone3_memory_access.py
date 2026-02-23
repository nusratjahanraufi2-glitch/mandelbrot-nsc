"""
Milestone 3: Memory Access Patterns (Week 2)

Goal:
- Compare row-wise vs column-wise traversal speed in NumPy arrays
- Then repeat for a Fortran-ordered (column-major) array

From the slide:
1) A = np.random.rand(N, N) with N = 10000
2) Row sums in a Python loop:   np.sum(A[i, :])
3) Col sums in a Python loop:   np.sum(A[:, j])
4) Time both
5) Repeat with Af = np.asfortranarray(A)
"""

import time
import statistics
import numpy as np


def time_median(func, n_runs=3):
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    print(f"Median over {n_runs} runs: {med:.4f}s (min={min(times):.4f}, max={max(times):.4f})")
    return med, result


def row_loop_sums(A: np.ndarray) -> float:
    """Loop over rows: A[i, :]"""
    N = A.shape[0]
    total = 0.0
    for i in range(N):
        total += np.sum(A[i, :])
    return total


def col_loop_sums(A: np.ndarray) -> float:
    """Loop over columns: A[:, j]"""
    N = A.shape[1]
    total = 0.0
    for j in range(N):
        total += np.sum(A[:, j])
    return total


def main():
    N = 10_000  # as in the slide

    print("Creating A (C-order / row-major by default)...")
    A = np.random.rand(N, N)
    print("A flags:", "C_CONTIGUOUS =", A.flags["C_CONTIGUOUS"], ", F_CONTIGUOUS =", A.flags["F_CONTIGUOUS"])

    print("\n--- C-order array A ---")
    print("Row loop (np.sum(A[i, :]))")
    t_row_A, s1 = time_median(lambda: row_loop_sums(A), n_runs=3)

    print("Column loop (np.sum(A[:, j]))")
    t_col_A, s2 = time_median(lambda: col_loop_sums(A), n_runs=3)

    print(f"\nC-order comparison: col/row speed ratio = {t_col_A / t_row_A:.2f}x slower (typically > 1)")
    print(f"Sanity check totals (should be close): diff = {abs(s1 - s2):.6f}")

    print("\nCreating Af = np.asfortranarray(A) (Fortran-order / column-major)...")
    Af = np.asfortranarray(A)
    print("Af flags:", "C_CONTIGUOUS =", Af.flags["C_CONTIGUOUS"], ", F_CONTIGUOUS =", Af.flags["F_CONTIGUOUS"])

    print("\n--- Fortran-order array Af ---")
    print("Row loop (np.sum(Af[i, :]))")
    t_row_Af, s3 = time_median(lambda: row_loop_sums(Af), n_runs=3)

    print("Column loop (np.sum(Af[:, j]))")
    t_col_Af, s4 = time_median(lambda: col_loop_sums(Af), n_runs=3)

    print(f"\nFortran-order comparison: row/col speed ratio = {t_row_Af / t_col_Af:.2f}x slower (typically > 1)")
    print(f"Sanity check totals (should be close): diff = {abs(s3 - s4):.6f}")

    print("\nWhat you should observe (and explain):")
    print("- In C-order (row-major), ROW loop is faster than COLUMN loop.")
    print("- In Fortran-order (column-major), COLUMN loop becomes faster than ROW loop.")
    print("- Reason: stride-1 contiguous access uses cache lines efficiently.")


if __name__ == "__main__":
    main()