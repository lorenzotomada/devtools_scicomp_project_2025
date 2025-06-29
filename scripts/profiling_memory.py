from pyclassify import Lanczos_PRO
from pyclassify.utils import (
    read_config,
    make_symmetric,
    profile_numpy_eigvals,
    profile_scipy_eigvals,
    poisson_2d_structure,
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
from mpi4py import MPI


seed = 8422
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
n_procs = kwargs["n_processes"]
plot = kwargs["plot"]

# Now we build the matrix on rank 0
# It is a scipy sparse matrix with the structure of a 2D Poisson problem matrix obtained using finite differences
if rank == 0:
    A = poisson_2d_structure(dim)
    A_np = A.toarray()
else:
    A_np = None

A_np = comm.bcast(A_np, root=0)


# On rank 0, we use the Lanczos method
# We actually call it twice: the first time to ensure that the function is JIT-compiled by Numba, the second one for memory profiling
if rank == 0:
    print("Precompiling Lanczos...")
    Q, diag, off_diag = Lanczos_PRO(A_np, np.ones_like(np.diag(A_np)) * 1.0)
    print("Done. Now reducing using Lanczos...")
    gc.collect()
    proc = psutil.Process()
    mem_before_lanczos = proc.memory_info().rss / 1024 / 1024  # MB

    Q, diag, off_diag = Lanczos_PRO(A_np, np.ones_like(np.diag(A_np)) * 1.0)

    gc.collect()
    mem_after_lanczos = proc.memory_info().rss / 1024 / 1024  # MB
    delta_mem_lanczos = mem_after_lanczos - mem_before_lanczos
    print("Done. Now computing eigenvalues...")
else:
    diag = off_diag = None

diag = comm.bcast(diag, root=0)
off_diag = comm.bcast(off_diag, root=0)

gc.collect()
proc = psutil.Process()
mem_before = proc.memory_info().rss / 1024 / 1024  # MB

eigvals, eigvecs = parallel_tridiag_eigen(
    diag, off_diag, comm=comm, min_size=1, tol_factor=1e-10
)

gc.collect()
mem_after = proc.memory_info().rss / 1024 / 1024
delta_mem = mem_after - mem_before

total_mem_children = comm.reduce(delta_mem, op=MPI.SUM, root=0)

if rank == 0:
    total_mem_all = delta_mem_lanczos
    print("Eigenvalues computed.")
    process = psutil.Process()

    print(f"Total memory across all processes: {total_mem_all:.2f} MB")

    mem_np = profile_numpy_eigvals(A_np)
    print(f"NumPy eig memory usage: {mem_np:.2f} MB")

    mem_sp = profile_scipy_eigvals(A_np)
    print(f"SciPy eig memory usage: {mem_sp:.2f} MB")

    os.makedirs("logs", exist_ok=True)
    log_file = "logs/memory_profile.csv"
    fieldnames = [
        "matrix_size",
        "n_processes",
        "mem_lanzos_mb",
        "mem_tridiag_mb",
        "mem_total_mb",
        "mem_numpy_mb",
        "mem_scipy_mb",
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
                "mem_lanzos_mb": round(delta_mem_lanczos, 2),
                "mem_tridiag_mb": round(total_mem_children, 2),
                "mem_total_mb": round(total_mem_all, 2),
                "mem_numpy_mb": round(mem_np, 2),
                "mem_scipy_mb": round(mem_sp, 2),
            }
        )

    if plot:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv("logs/memory_profile.csv")
        nproc_values = sorted(df["n_processes"].unique())

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

        plt.savefig("logs/mem_vs_size_all_methods.png", bbox_inches="tight")
        plt.show()
