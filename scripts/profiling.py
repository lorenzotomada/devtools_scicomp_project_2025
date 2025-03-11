import cupyx.scipy.sparse as cpsp
import cupy as cp
from pyclassify import (
    eigenvalues_np,
    eigenvalues_sp,
    eigenvalues_cp,
    power_method,
    power_method_numba,
    power_method_cp,
    QR,
    QR_cp,
)
from pyclassify.utils import make_symmetric, read_config
import numpy as np
import scipy.sparse as sp
import random
import argparse
import cProfile
import os
import csv


seed = 8422
random.seed(seed)
cp.random.seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False, help="config file:")

args = parser.parse_args()
filename = (
    args.config if args.config else "./experiments/config"
)  # automatic choice if no argument is passed

kwargs = read_config(filename)
dim = kwargs["dim"]
density = kwargs["density"]
tol = kwargs["tol"]
max_iter = kwargs["max_iter"]


matrix = sp.random(dim, dim, density=density, format="coo")
matrix = make_symmetric(matrix)
cp_symm_matrix = cpsp.csr_matrix(matrix)


logs = "./logs"
iteration_factor = 300


def profile_with_cprofile(func_name, func, *args, **kwargs):
    log_file = "./logs/timings.csv"
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        profile_output = cProfile.Profile()
        profile_output.enable()

        result = func(*args, **kwargs)

        profile_output.disable()

        stats = profile_output.getstats()
        total_time = sum(stat.totaltime for stat in stats)

        print(f"{func_name}: {total_time:.6f} s")

        writer.writerow([func_name, dim, total_time])

    return result


profile_with_cprofile(
    "eigenvalues_np", eigenvalues_np, matrix.toarray(), symmetric=True
)
profile_with_cprofile("eigenvalues_sp", eigenvalues_sp, matrix, symmetric=True)
profile_with_cprofile("power_method", power_method, matrix)
profile_with_cprofile("power_method_numba", power_method_numba, matrix.toarray())
profile_with_cprofile("QR", QR, matrix.toarray(), max_iter=iteration_factor * dim)


def profile_with_cupy_profiler(func_name, func, *args, **kwargs):
    log_file = "./logs/timings.csv"
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        result = func(*args, **kwargs)
        end.record()
        end.synchronize()

        result = func(*args, **kwargs)
        elapsed_time = cp.cuda.get_elapsed_time(start, end) / 1000
        print(f"{func_name}: {elapsed_time:.6f} s")

        writer.writerow([func_name, dim, elapsed_time])

    return result


profile_with_cupy_profiler("eigenvalues_cp", eigenvalues_cp, cp_symm_matrix)
profile_with_cupy_profiler("power_method_cp", power_method_cp, cp_symm_matrix)
profile_with_cupy_profiler(
    "QR_cp",
    QR_cp,
    cp_symm_matrix,
    q0=cp.random.rand(dim),
    tol=1e-3,
    max_iter=iteration_factor * dim,
)
