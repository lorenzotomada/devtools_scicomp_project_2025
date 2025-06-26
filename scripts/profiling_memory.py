from pyclassify import Lanczos_PRO
from pyclassify.utils import read_config, make_symmetric, profile_numpy_eigvals, profile_scipy_eigvals, poisson_2d_structure
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
sys.path.append('scripts')
#from mpi_running import compute_eigvals


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

    print('Data recieved!')
    return eigvals, eigvecs, delta_t, total_mem_children


def compute_eigvals(A, n_procs):
    print('Reducing using Lanczos')
    Q, diag, off_diag = Lanczos_PRO(A_np, np.ones_like(np.diag(A_np)) * 1.0)

    print('Done. Now computing eigenvalues.')
    eigvals, eigvecs, delta_t, total_mem_children = parallel_eig(diag, off_diag, n_procs)

    print("Eigenvalues computed")
    return eigvals, eigvecs, delta_t, total_mem_children


A = poisson_2d_structure(dim)
A_np = A.toarray()

Q, diag, off_diag = Lanczos_PRO(A_np, np.ones_like(np.diag(A_np)) * 1.0) # To compile using numba

gc.collect()
process = psutil.Process()
mem_before = process.memory_info().rss / 1024 / 1024

eigvals, eigvecs, delta_t, total_mem_children = compute_eigvals(A_np, n_procs)

gc.collect()
mem_after = process.memory_info().rss / 1024 / 1024
delta_mem_parent = mem_after - mem_before

total_mem_all = delta_mem_parent + total_mem_children

print(f"[Parent] Memory used by parent: {delta_mem_parent:.2f} MB")
print(f"[Parent] Memory used by all child processes: {total_mem_children:.2f} MB")
print(f"[Parent] Total memory across all processes: {total_mem_all:.2f} MB")

mem_np = profile_numpy_eigvals(A_np)
print(f"NumPy eigh memory usage: {mem_np:.2f} MB")

mem_sp = profile_scipy_eigvals(A_np)
print(f"SciPy eigh memory usage: {mem_sp:.2f} MB")

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

    df = pd.read_csv("logs/memory_profile.csv")

    # Plot 1: Total memory vs number of processes
    plt.figure(figsize=(8, 5))
    for size in sorted(df["matrix_size"].unique()):
        subset = df[df["matrix_size"] == size]
        plt.plot(
            subset["n_processes"],
            subset["mem_total_mb"],
            marker="o",
            label=f"Size {size}",
        )
    plt.xlabel("Number of Processes")
    plt.ylabel("Total Memory (MB)")
    plt.title("Total Memory vs. Number of Processes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/mem_total_vs_processes.png")
    plt.show()

    # Plot 2: Separate memory (parent & children) vs number of processes
    plt.figure(figsize=(8, 5))
    for size in sorted(df["matrix_size"].unique()):
        subset = df[df["matrix_size"] == size]
        plt.plot(
            subset["n_processes"],
            subset["mem_parent_mb"],
            marker="s",
            linestyle="-",
            label=f"Parent (size {size})",
        )
        plt.plot(
            subset["n_processes"],
            subset["mem_children_mb"],
            marker="^",
            linestyle="--",
            label=f"Children (size {size})",
        )
    plt.xlabel("Number of Processes")
    plt.ylabel("Memory (MB)")
    plt.title("Parent vs. Children Memory vs. Number of Processes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/mem_parent_children_vs_processes.png")
    plt.show()

    # Plot 3: Total memory vs matrix size (for fixed number of processes)
    fixed_nprocs = df["n_processes"].mode()[0]
    subset = df[df["n_processes"] == fixed_nprocs]
    plt.figure(figsize=(8, 5))
    plt.plot(
        subset["matrix_size"],
        subset["mem_total_mb"],
        marker="o",
        label="Parent + Children",
    )
    plt.plot(
        subset["matrix_size"],
        subset["mem_numpy_mb"],
        marker="x",
        linestyle="--",
        label="NumPy",
    )
    plt.plot(
        subset["matrix_size"],
        subset["mem_scipy_mb"],
        marker="^",
        linestyle=":",
        label="SciPy",
    )
    plt.xlabel("Matrix Size")
    plt.ylabel("Memory (MB)")
    plt.title(f"Memory vs. Matrix Size (n_procs={fixed_nprocs})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/mem_vs_size.png")
    plt.show()

    # Plot 4: Separate parent/children memory vs matrix size (fixed nprocs)
    plt.figure(figsize=(8, 5))
    plt.plot(subset["matrix_size"], subset["mem_parent_mb"], marker="s", label="Parent")
    plt.plot(
        subset["matrix_size"], subset["mem_children_mb"], marker="^", label="Children"
    )
    plt.xlabel("Matrix Size")
    plt.ylabel("Memory (MB)")
    plt.title(f"Parent vs. Children Memory vs. Matrix Size (n_procs={fixed_nprocs})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/mem_parent_children_vs_size.png")
    plt.show()

    # Plot 5: Memory usage vs matrix size for all methods
    plt.figure(figsize=(10, 6))

    # NumPy and SciPy (plotted once, same for all n_processes)
    numpy_avg = df.groupby("matrix_size")["mem_numpy_mb"].mean()
    scipy_avg = df.groupby("matrix_size")["mem_scipy_mb"].mean()
    plt.plot(
        numpy_avg.index,
        numpy_avg.values,
        marker="x",
        linestyle="--",
        color="gray",
        label="NumPy",
    )
    plt.plot(
        scipy_avg.index,
        scipy_avg.values,
        marker="^",
        linestyle=":",
        color="black",
        label="SciPy",
    )

    # Your method for each n_processes value
    for nproc in sorted(df["n_processes"].unique()):
        subset = df[df["n_processes"] == nproc]
        label = f"My method ({nproc} proc{'s' if nproc > 1 else ''})"
        plt.plot(subset["matrix_size"], subset["mem_total_mb"], marker="o", label=label)

    plt.xlabel("Matrix Size")
    plt.ylabel("Memory (MB)")
    plt.title("Memory Usage vs. Matrix Size for Different Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/mem_comparison_vs_size.png")
    plt.show()