import random
import time
import dask
from dask import delayed


def monte_carlo_chunk(n_samples):
    inside = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1:
            inside += 1
    return inside


if __name__ == "__main__":
    total = 1_000_000
    n_chunks = 8
    samples = total // n_chunks

    # 1) Serial baseline
    t0 = time.perf_counter()
    results_serial = [monte_carlo_chunk(samples) for _ in range(n_chunks)]
    t_serial = time.perf_counter() - t0
    pi_serial = 4 * sum(results_serial) / total
    print(f"Serial: {t_serial:.3f} s, pi = {pi_serial:.6f}")

    # 2) Dask delayed: build task graph only, no execution yet
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]

    print("Delayed object example:")
    print(tasks[0])

    # 3) Visualize task graph
    # May require: conda install python-graphviz
    dask.visualize(*tasks, filename="task_graph.png")

    # 4) Execute all tasks
    t0 = time.perf_counter()
    results_dask = dask.compute(*tasks)
    t_dask = time.perf_counter() - t0
    pi_dask = 4 * sum(results_dask) / total
    print(f"Dask:   {t_dask:.3f} s, pi = {pi_dask:.6f}")