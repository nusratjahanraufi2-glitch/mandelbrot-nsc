import time
import numpy as np
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster


@delayed
def generate(seed, n):
    time.sleep(0.3)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


@delayed
def chunk_max(data):
    time.sleep(0.2)
    return float(np.max(np.abs(data)))


@delayed
def global_max(maxima):
    time.sleep(0.2)
    return max(maxima)


@delayed
def normalise(data, g):
    time.sleep(0.3)
    return data / g


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    print(f"Dashboard: {client.dashboard_link}")
    print("Open the dashboard in your browser.")
    input("Press Enter after opening the dashboard...")

    # Stage 1a: generate chunks
    chunks = [generate(i, 50_000) for i in range(8)]

    # Stage 1b: compute per-chunk maxima
    maxima = [chunk_max(c) for c in chunks]

    # Stage 2: fan-in to one task
    gmax = global_max(maxima)

    # Stage 3: fan-out normalization
    normed = [normalise(c, gmax) for c in chunks]

    # Visualize task graph
    dask.visualize(*normed, filename="task_graph_pipeline", format="png")

    print("Task graph saved as task_graph_pipeline.png")
    input("Press Enter to start computation and watch Task Stream...")

    t0 = time.perf_counter()
    results = dask.compute(*normed)
    wall_time = time.perf_counter() - t0

    print(f"Wall time: {wall_time:.2f} s")
    print(f"Number of normalized chunks: {len(results)}")
    print(f"Shape of first chunk: {results[0].shape}")

    input("Press Enter to close the cluster and exit...")
    client.close()
    cluster.close()