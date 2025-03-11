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
from pyclassify.utils import (
    make_symmetric,
    read_config,
    profile_with_cprofile,
    profile_with_cupy_profiler,
)
import numpy as np
import scipy.sparse as sp
import scipy
import random
import argparse


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


eigenvals = np.arange(1, dim + 1)
A = np.diag(eigenvals)
U = scipy.stats.ortho_group.rvs(dim)
A = U @ A @ U.T
A = make_symmetric(A)
A = sp.csr_matrix(A)
A_cp = cpsp.csr_matrix(A)


log_file = "./logs/timings.csv"
iteration_factor = 300


profile_with_cprofile(
    log_file, dim, "eigenvalues_np", eigenvalues_np, A.toarray(), symmetric=True
)
profile_with_cprofile(
    log_file, dim, "eigenvalues_sp", eigenvalues_sp, A, symmetric=True
)
profile_with_cprofile(log_file, dim, "power_method", power_method, A)
profile_with_cprofile(
    log_file, dim, "power_method_numba", power_method_numba, A.toarray()
)
profile_with_cprofile(
    log_file, dim, "QR", QR, A.toarray(), max_iter=iteration_factor * dim
)


profile_with_cupy_profiler(log_file, dim, "eigenvalues_cp", eigenvalues_cp, A_cp)
profile_with_cupy_profiler(log_file, dim, "power_method_cp", power_method_cp, A_cp)
profile_with_cupy_profiler(
    log_file,
    dim,
    "QR_cp",
    QR_cp,
    A_cp,
    q0=cp.random.rand(dim),
    tol=1e-3,
    max_iter=iteration_factor * dim,
)
