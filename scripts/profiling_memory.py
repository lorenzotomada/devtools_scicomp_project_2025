from pyclassify import parallel_tridiag_eigen, Lanczos_PRO
from pyclassify.utils import read_config, make_symmetric
from time import time
import numpy as np
import argparse
from mpi4py import MPI
import sys
import scipy
import scipy.sparse as sp
import psutil
import gc
import os
import csv


# Seed for reproducibility
seed = 8422
np.random.seed(seed)


def parallel_eig(d, off_d, nprocs):
    print("inside parallel_eig")
    comm = MPI.COMM_SELF.Spawn(
        sys.executable, args=["scripts/run.py"], maxprocs=nprocs
    )
    print("sending")
    comm.send(d, dest=0, tag=11)
    comm.send(off_d, dest=0, tag=12)
    print("Sent data to child, waiting for results...")
    sys.stdout.flush()

    eigvals = comm.recv(source=0, tag=22)
    eigvecs = comm.recv(source=0, tag=23)
    delta_t = comm.recv(source=0, tag=24)
    total_mem_children = comm.recv(source=0, tag=25)

    comm.Disconnect()
    return eigvals, eigvecs, delta_t, total_mem_children


def compute_eigvals(A, n_procs):
    #A = A.toarray()
    initial_guess = np.random.rand(A.shape[0])
    if initial_guess[0] == 0:
        initial_guess[0] += 1
    Q, diag, off_diag = Lanczos_PRO(A, np.ones_like(np.diag(A)) * 1.)
    eigvals, _, __, total_mem_children = parallel_eig(diag, off_diag, n_procs)
    print("Eigenvalues computed")
    return total_mem_children


def profile_numpy_eigvals(A):
    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    # NumPy symmetric eig solver
    eigvals, eigvecs = np.linalg.eigh(A)

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    delta_mem = mem_after - mem_before
    return delta_mem


def profile_scipy_eigvals(A):
    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    # SciPy symmetric eig solver
    eigvals, eigvecs = scipy.linalg.eigh(A)

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    delta_mem = mem_after - mem_before
    return delta_mem


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False, help="config file:")
args = parser.parse_args()
config_file = args.config if args.config else "./experiments/config"

kwargs = read_config(config_file)
dim = kwargs["dim"]
density = kwargs["density"]
n_procs = kwargs["n_processes"]
plot = kwargs["plot"]

A = sp.random(dim, dim, density=density, format="csr")
A = make_symmetric(A)

eig = np.arange(1, dim + 1)
A = np.diag(eig)
U = scipy.stats.ortho_group.rvs(dim)

A = U @ A @ U.T
A = make_symmetric(A)
A_sp = sp.csr_matrix(A)

gc.collect()
process = psutil.Process()
mem_before = process.memory_info().rss / 1024 / 1024

total_mem_children = compute_eigvals(A, n_procs)

gc.collect()
mem_after = process.memory_info().rss / 1024 / 1024
delta_mem_parent = mem_after - mem_before

total_mem_all = delta_mem_parent + total_mem_children

print(f"[Parent] Memory used by parent: {delta_mem_parent:.2f} MB")
print(f"[Parent] Memory used by all child processes: {total_mem_children:.2f} MB")
print(f"[Parent] Total memory across all processes: {total_mem_all:.2f} MB")

mem_np = profile_numpy_eigvals(A)
print(f"NumPy eigh memory usage: {mem_np:.2f} MB")

mem_sp = profile_scipy_eigvals(A)
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
    "mem_scipy_mb"
]

write_header = not os.path.exists(log_file)

with open(log_file, mode="a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow({
        "matrix_size": dim,
        "n_processes": n_procs,
        "mem_parent_mb": round(delta_mem_parent, 2),
        "mem_children_mb": round(total_mem_children, 2),
        "mem_total_mb": round(total_mem_all, 2),
        "mem_numpy_mb": round(mem_np, 2),
        "mem_scipy_mb": round(mem_sp, 2)
    })

if plot:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("logs/memory_profile.csv")

    # Plot 1: Total memory vs number of processes
    plt.figure(figsize=(8, 5))
    for size in sorted(df["matrix_size"].unique()):
        subset = df[df["matrix_size"] == size]
        plt.plot(subset["n_processes"], subset["mem_total_mb"], marker="o", label=f"Size {size}")
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
        plt.plot(subset["n_processes"], subset["mem_parent_mb"], marker="s", linestyle="-", label=f"Parent (size {size})")
        plt.plot(subset["n_processes"], subset["mem_children_mb"], marker="^", linestyle="--", label=f"Children (size {size})")
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
    plt.plot(subset["matrix_size"], subset["mem_total_mb"], marker="o", label="Parent + Children")
    plt.plot(subset["matrix_size"], subset["mem_numpy_mb"], marker="x", linestyle="--", label="NumPy")
    plt.plot(subset["matrix_size"], subset["mem_scipy_mb"], marker="^", linestyle=":", label="SciPy")
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
    plt.plot(subset["matrix_size"], subset["mem_children_mb"], marker="^", label="Children")
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
    plt.plot(numpy_avg.index, numpy_avg.values, marker="x", linestyle="--", color="gray", label="NumPy")
    plt.plot(scipy_avg.index, scipy_avg.values, marker="^", linestyle=":", color="black", label="SciPy")

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
