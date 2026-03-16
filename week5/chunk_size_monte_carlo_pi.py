from multiprocessing import Pool
import random
import time
import os


def monte_carlo_chunk(num_samples):
    inside = 0

    for _ in range(num_samples):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            inside += 1

    return inside


def test_granularity(total_work, chunk_size, n_proc):
    n_chunks = total_work // chunk_size
    tasks = [chunk_size] * n_chunks

    t0 = time.perf_counter()

    if n_proc == 1:
        results = [monte_carlo_chunk(s) for s in tasks]
    else:
        with Pool(processes=n_proc) as pool:
            results = pool.map(monte_carlo_chunk, tasks)

    elapsed = time.perf_counter() - t0
    pi_est = 4 * sum(results) / total_work

    return elapsed, pi_est


if __name__ == "__main__":
    total_work = 1_000_000
    n_proc = os.cpu_count() // 2

    chunk_sizes = [10, 100, 1000, 10000, 100000, 1000000]

    print(f"{'L':>12} | {'serial (s)':>12} | {'parallel (s)':>12}")

    for L in chunk_sizes:
        t_ser, _ = test_granularity(total_work, L, 1)
        t_par, pi = test_granularity(total_work, L, n_proc)

        print(f"{L:12d} | {t_ser:12.4f} | {t_par:12.4f} | pi={pi:.4f}")