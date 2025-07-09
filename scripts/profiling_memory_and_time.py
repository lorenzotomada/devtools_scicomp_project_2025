from pyclassify import Lanczos_PRO
from pyclassify.utils import (
    read_config,
    make_symmetric,
    profile_numpy_eigvals,
    profile_scipy_eigvals,
)
from pyclassify.parallel_tridiag_eigen import parallel_tridiag_eigen

import numpy as np
import scipy
import argparse
import scipy.sparse as sp
import psutil
import gc
import os
import csv
import sys
from time import time
from mpi4py import MPI

from line_profiler import LineProfiler

profile = LineProfiler()
profile.add_function(parallel_tridiag_eigen)


seed = 1000
np.random.seed(seed)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="config file:")
    args = parser.parse_args()
    config_file = args.config if args.config else "experiments/config"
    kwargs = read_config(config_file)
    dim = kwargs["dim"]
    density = kwargs["density"]
    n_procs = kwargs["n_processes"]
    plot = kwargs["plot"]
else:
    kwargs = None

# Broadcast config to all ranks
kwargs = comm.bcast(kwargs, root=0)
dim = kwargs["dim"]
density = kwargs["density"]
n_procs = size  # kwargs["n_processes"]
plot = kwargs["plot"]

# Now we build the matrix on rank 0
if rank == 0:
    eig = np.arange(1, dim + 1)
    A = np.diag(eig)
    U = scipy.stats.ortho_group.rvs(dim)

    A = U @ A @ U.T
    A = make_symmetric(A)
    A_np = A
else:
    A_np = None

A_np = comm.bcast(A_np, root=0)


# On rank 0, we use the Lanczos method.
# We actually call it twice: the first time to ensure that the function is JIT-compiled by Numba, the second one for memory profiling
if rank == 0:
    print("Precompiling Lanczos...")

    Q, diag, off_diag = Lanczos_PRO(A_np, np.ones_like(np.diag(A_np)) * 1.0)

    print("Done. Now reducing using Lanczos...")

    gc.collect()
    proc = psutil.Process()
    mem_before_lanczos = proc.memory_info().rss / 1024 / 1024  # MB
    begin_lanczos = time()

    Q, diag, off_diag = Lanczos_PRO(A_np, np.ones_like(np.diag(A_np)) * 1.0)

    end_lanczos = time()
    gc.collect()
    mem_after_lanczos = proc.memory_info().rss / 1024 / 1024  # MB
    delta_mem_lanczos = mem_after_lanczos - mem_before_lanczos
    delta_t_lanczos = end_lanczos - begin_lanczos

    print("Done. Now computing eigenvalues...")
else:
    diag = off_diag = None

# Now we broadcast diag and off_diag to all other ranks so we can use parallel_tridiag_eigen
diag = comm.bcast(diag, root=0)
off_diag = comm.bcast(off_diag, root=0)

gc.collect()
proc = psutil.Process()
mem_before = proc.memory_info().rss / 1024 / 1024  # MB
time_before_parallel = time()
profile.enable()
eigvals, eigvecs = parallel_tridiag_eigen(
    diag, off_diag, comm=comm, min_size=1, tol_factor=1e-10
)
profile.disable()
profile_filename = f"Profiling_files/profile_rank{comm.Get_rank()}.lprof"
with open(profile_filename, "w") as f:
    profile.print_stats(stream=f)
time_after_parallel = time()
gc.collect()
mem_after = proc.memory_info().rss / 1024 / 1024
delta_mem = mem_after - mem_before
delta_t_parallel = time_after_parallel - time_before_parallel

total_mem_children = comm.reduce(delta_mem, op=MPI.SUM, root=0)
# total_time_children = comm.reduce(delta_t_parallel, op=MPI.SUM, root=0) #only if we want the sum!
total_time_children = delta_t_parallel

# Collect the information across all ranks
if rank == 0:
    total_mem_all = delta_mem_lanczos + total_mem_children
    total_time_all = delta_t_lanczos + total_time_children
    print("Eigenvalues computed.")
    process = psutil.Process()

    print(f"[D&I] Total memory across all processes: {total_mem_all:.4f} MB")
    print(
        f"[D&I] Total time (rank 0, which also performs Lanczos): {total_time_all:.4f} s"
    )
    # We also profile numpy and scipy memory consumption
    mem_np, time_np = profile_numpy_eigvals(A_np)
    print(f"[NumPy] eig memory usage: {mem_np:.4f} MB")
    print(f"[NumPy] eig total time: {time_np:.4f} s")

    mem_sp, time_sp = profile_scipy_eigvals(A_np)
    print(f"[SciPy] eig memory usage: {mem_sp:.4f} MB")
    print(f"[SciPy] eig total time: {time_sp:.4f} s")

    # Save to the logs folder
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/profile.csv"
    fieldnames = [
        "matrix_size",
        "n_processes",
        "mem_lanczos_mb",
        "mem_tridiag_mb",
        "mem_total_mb",
        "mem_numpy_mb",
        "mem_scipy_mb",
        "time_lanczos",
        "time_tridiag",
        "time_total",
        "time_numpy",
        "time_scipy",
    ]

    write_header = not os.path.exists(log_file)
    with open(log_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "matrix_size": dim,
                "n_processes": size,
                "mem_lanczos_mb": round(delta_mem_lanczos, 2),
                "mem_tridiag_mb": round(total_mem_children, 2),
                "mem_total_mb": round(total_mem_all, 2),
                "mem_numpy_mb": round(mem_np, 2),
                "mem_scipy_mb": round(mem_sp, 2),
                "time_lanczos": round(delta_t_lanczos, 2),
                "time_tridiag": round(total_time_children, 2),
                "time_total": round(total_time_all, 2),
                "time_numpy": round(time_np, 2),
                "time_scipy": round(time_sp, 2),
            }
        )

    if plot:
        # We only plot if all the runs have been done already. In this way, we get a complete memory usage graph.

        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv("logs/profile.csv")
        nproc_values = sorted(df["n_processes"].unique())

        # First we plot the memoy usage, then the execution time
        plt.figure(figsize=(10, 6))

        numpy_avg = df.groupby("matrix_size")["mem_numpy_mb"].mean()
        plt.plot(
            numpy_avg.index,
            numpy_avg.values,
            color="green",
            marker="x",
            linestyle="--",
            label="NumPy",
        )

        scipy_avg = df.groupby("matrix_size")["mem_scipy_mb"].mean()
        plt.plot(
            scipy_avg.index,
            scipy_avg.values,
            color="red",
            marker="^",
            linestyle=":",
            label="SciPy",
        )

        for nproc in nproc_values:
            subset = df[df["n_processes"] == nproc].sort_values("matrix_size")
            label = f"Divide et impera ({nproc} proc{'s' if nproc > 1 else ''})"
            plt.plot(
                subset["matrix_size"],
                subset["mem_total_mb"],
                marker="o",
                linestyle="-",
                label=label,
            )

        plt.xlabel("Matrix size")
        plt.ylabel("Total memory (MB)")
        plt.xscale("log")
        plt.title("Memory usage vs. Matrix size")
        plt.grid(True)
        plt.tight_layout()

        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            title="Method",
        )
        plt.subplots_adjust(right=0.75)

        plt.savefig("logs/memory_profiling.png", bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(10, 6))

        numpy_avg = df.groupby("matrix_size")["time_numpy"].mean()
        plt.plot(
            numpy_avg.index,
            numpy_avg.values,
            color="green",
            marker="x",
            linestyle="--",
            label="NumPy",
        )

        scipy_avg = df.groupby("matrix_size")["time_scipy"].mean()
        plt.plot(
            scipy_avg.index,
            scipy_avg.values,
            color="red",
            marker="^",
            linestyle=":",
            label="SciPy",
        )

        for nproc in nproc_values:
            subset = df[df["n_processes"] == nproc].sort_values("matrix_size")
            label = f"Divide et impera ({nproc} proc{'s' if nproc > 1 else ''})"
            plt.plot(
                subset["matrix_size"],
                subset["time_total"],
                marker="o",
                linestyle="-",
                label=label,
            )

        plt.xlabel("Matrix size")
        plt.ylabel("Total time (s)")
        plt.xscale("log")
        plt.title("Execution time vs. Matrix size")
        plt.grid(True)
        plt.tight_layout()

        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            title="Method",
        )
        plt.subplots_adjust(right=0.75)

        plt.savefig("logs/time_profiling.png", bbox_inches="tight")
        plt.show()
