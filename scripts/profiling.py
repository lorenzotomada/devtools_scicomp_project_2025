from pyclassify import power_method, power_method_numba
from pyclassify.utils import make_symmetric, read_config
from pyclassify.profiling_MPI import mpi_profiled, get_memory_usage_mb, profile_serial
import numpy as np
import scipy.sparse as sp
import random
import argparse
import os
import time
import pandas as pd
from mpi4py import MPI


# Seed for reproducibility
seed = 8422
random.seed(seed)
np.random.seed(seed)

# Some MPI info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()


# Here we parse the arguments. We provide a default value, but the user is free to chose another config file.
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False, help="config file:")
args = parser.parse_args()
config_file = args.config if args.config else "./experiments/config_profiling"


# Now read the config only if the rank is 0. We will broadcast the info to all other ranks.
if rank == 0:
    kwargs = read_config(config_file)
    dim = kwargs["dim"]
    density = kwargs["density"]
else:
    dim = None
    density = None

dim = comm.bcast(dim, root=0)
density = comm.bcast(density, root=0)


# Generate the data
if rank == 0:
    A = sp.random(dim, dim, density=density, format="csr")
    A = make_symmetric(A)
else:
    A = None

A = comm.bcast(A, root=0)


# Now we start profiling. Notice that the only function that requires MPI is the one that is not profiled within a 'if rank==0' statement.

# @mpi_profiled
# def profiled_divide_et_impera(A):
#    from pyclassify import divide_et_impera  # avoid circular import
#    return divide_et_impera(A)

results = {}


if rank == 0:
    _ = profile_serial(power_method_numba, A.toarray())
    results["power_method"] = profile_serial(power_method, A)
    results["power_method_numba"] = profile_serial(power_method_numba, A.toarray())
    # results["QR"] = profile_serial(QR, A)

# mpi_result_QR = profiled_QR(A)
# mpi_result_divide = profiled_divide_et_impera(A)

# if rank == 0:
#    results["QR"] = mpi_result_QR
#    results["divide_et_impera"] = mpi_result_divide


# Now we just save to CSV.
if rank == 0:
    os.makedirs("logs", exist_ok=True)
    mem_csv = "logs/memory.csv"
    time_csv = "logs/time.csv"

    mem_row = {
        "matrix_size": dim,
        "density": density,
        "num_procs": n_procs,
        **{key: results[key]["memory_total"] for key in results},
    }

    time_row = {
        "matrix_size": dim,
        "density": density,
        "num_procs": n_procs,
        **{key: results[key]["time"] for key in results},
    }

    def append_or_create_csv(path, row, columns):
        """
        This helper function just decides whether to append to an existing CSV or to create a new one.
        """
        if not os.path.exists(path):
            pd.DataFrame([row]).to_csv(path, index=False, columns=columns)
        else:
            df = pd.read_csv(path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(path, index=False, columns=columns)

    method_names = list(results.keys())
    base_cols = ["matrix_size", "density", "num_procs"]
    all_columns = base_cols + method_names

    append_or_create_csv(mem_csv, mem_row, all_columns)
    append_or_create_csv(time_csv, time_row, all_columns)

    print("Done! The results have been saved to logs/memory.csv and logs/time.csv")
