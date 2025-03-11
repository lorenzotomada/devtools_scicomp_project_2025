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
import cupyx.profiler as profiler
import os
import csv


random.seed(226)
cp.random.seed(226)
np.random.seed(226)


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


matrix = sp.random(dim, dim, density=density, format="csr")
matrix = make_symmetric(matrix)
cp_symm_matrix = cpsp.csr_matrix(matrix)


logs = "./logs"
iteration_factor = 300


cProfile.run(
    "eigenvalues_np(matrix.toarray(), symmetric=True)",
    os.path.join(logs, f"eigenvalues_np_{dim}.prof"),
)
cProfile.run(
    "eigenvalues_np(matrix, symmetric=True)",
    os.path.join(logs, f"eigenvalues_sp_{dim}.prof"),
)
cProfile.run("power_method(matrix)", os.path.join(logs, f"power_method_{dim}.prof"))
cProfile.run(
    "power_method(matrix.toarray())",
    os.path.join(logs, f"power_method_numba{dim}.prof"),
)
cProfile.run(
    "QR(matrix.toarray(), max_iter=iteration_factor*d)",
    os.path.join(logs, f"QR_{dim}.prof"),
)


def profile_function(func_name, func, *args, **kwargs):
    cupyx.profiler.start()
    result = func(*args, **kwargs)
    cupyx.profiler.stop()

    profiler_data = cupyx.profiler.get_profiler()

    output_file = os.path.join(logs, f"{func_name}_{dim}.prof")
    with open(output_file, "w") as f:
        f.write(str(profiler_data))

    elapsed_time = profiler_data[0]["time"]
    print(f"{func_name}: {elapsed_time/1000} s")
    return result


profile_function("eigenvalues_cp", eigenvalues_cp, cp_symm_matrix)
profile_function("power_method_cp", power_method_cp, cp_symm_matrix)
profile_function("QR_cp", QR_cp, cp_symm_matrix, max_iter=iteration_factor * d)
