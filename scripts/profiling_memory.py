from pyclassify import Lanczos_PRO
from pyclassify.utils import (
    read_config,
    make_symmetric,
    profile_numpy_eigvals,
    profile_scipy_eigvals,
    poisson_2d_structure,
)
import numpy as np
import scipy
import argparse
from mpi4py import MPI
import scipy
import scipy.sparse as sp
import psutil
import gc
import os
import csv
import sys

sys.path.append("scripts")
# from mpi_running import compute_eigvals


# Seed for reproducibility
seed = 8422
np.random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False, help="config file:")
args = parser.parse_args()
config_file = args.config if args.config else "experiments/config"

kwargs = read_config(config_file)
dim = kwargs["dim"]
density = kwargs["density"]
n_procs = kwargs["n_processes"]
plot = kwargs["plot"]


def parallel_eig(diag, off_diag, nprocs):
    print("Spawning a communicator")
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=["scripts/run.py"], maxprocs=nprocs)

    print("Sending data to children")
    comm.send(diag, dest=0, tag=11)
    comm.send(off_diag, dest=0, tag=12)

    print("Waiting for results...")
    sys.stdout.flush()

    eigvals = comm.recv(source=0, tag=22)
    eigvecs = comm.recv(source=0, tag=23)
    delta_t = comm.recv(source=0, tag=24)
    total_mem_children = comm.recv(source=0, tag=25)
    comm.Disconnect()

    print("Data recieved!")
    return eigvals, eigvecs, delta_t, total_mem_children


def compute_eigvals(A, n_procs):
    print("Reducing using Lanczos")
    Q, diag, off_diag = Lanczos_PRO(A_np, np.ones_like(np.diag(A_np)) * 1.0)

    print("Done. Now computing eigenvalues.")
    eigvals, eigvecs, delta_t, total_mem_children = parallel_eig(
        diag, off_diag, n_procs
    )

    print("Eigenvalues computed")
    return eigvals, eigvecs, delta_t, total_mem_children


A = poisson_2d_structure(dim)
A_np = A.toarray()

Q, diag, off_diag = Lanczos_PRO(
    A_np, np.ones_like(np.diag(A_np)) * 1.0
)  # To compile using numba

gc.collect()
process = psutil.Process()
mem_before = process.memory_info().rss / 1024 / 1024

eigvals, eigvecs, delta_t, total_mem_children = compute_eigvals(A_np, n_procs)

gc.collect()
mem_after = process.memory_info().rss / 1024 / 1024
delta_mem_parent = mem_after - mem_before

total_mem_all = delta_mem_parent + total_mem_children

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
    "mem_parent_mb",
    "mem_children_mb",
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
            "n_processes": n_procs,
            "mem_parent_mb": round(delta_mem_parent, 2),
            "mem_children_mb": round(total_mem_children, 2),
            "mem_total_mb": round(total_mem_all, 2),
            "mem_numpy_mb": round(mem_np, 2),
            "mem_scipy_mb": round(mem_sp, 2),
        }
    )

if plot:
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.colors as mcolors
    import numpy as np

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
        label = f"Divide and Conquer ({nproc} proc{'s' if nproc > 1 else ''})"
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
    # plt.yscale("log")
    plt.title("Memory usage vs. Matrix size")
    plt.grid(True)
    plt.tight_layout()

    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, title="Method"
    )
    plt.subplots_adjust(right=0.75)

    plt.savefig("logs/mem_vs_size_all_methods.png", bbox_inches="tight")
    plt.show()
