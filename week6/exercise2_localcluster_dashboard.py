import random
import time
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster


def monte_carlo_chunk(n_samples):
    inside = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1:
            inside += 1
    return inside


def run_experiment(total, n_chunks):
    samples = total // n_chunks
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]

    t0 = time.perf_counter()
    results = dask.compute(*tasks)
    t_taken = time.perf_counter() - t0

    pi_est = 4 * sum(results) / total
    return t_taken, pi_est


if __name__ == "__main__":
    total = 1_000_000
    n_chunks = 8

    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    print(f"Dashboard: {client.dashboard_link}")
    print("Open this link in your browser now.")
    input("Press Enter after opening the dashboard...")

    t8, pi8 = run_experiment(total, n_chunks)
    print(f"8 workers -> time: {t8:.3f} s, pi = {pi8:.6f}")
    input("Check the dashboard. Press Enter to continue to 4 workers...")

    cluster.scale(4)
    client.wait_for_workers(4)
    t4, pi4 = run_experiment(total, n_chunks)
    print(f"4 workers -> time: {t4:.3f} s, pi = {pi4:.6f}")
    input("Check the dashboard. Press Enter to continue to 2 workers...")

    cluster.scale(2)
    client.wait_for_workers(2)
    t2, pi2 = run_experiment(total, n_chunks)
    print(f"2 workers -> time: {t2:.3f} s, pi = {pi2:.6f}")

    input("Finished. Press Enter to close the cluster and exit.")
    client.close()
    cluster.close()