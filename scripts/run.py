import cupyx.scipy.sparse as cpsp
import cupy as cp
from pyclassify import (
    eigenvalues_np,
    eigenvalues_sp,
    eigenvalues_cp,
    power_method,
    power_method_numba,
    power_method_cp,
)
from pyclassify.utils import make_symmetric, read_config
import numpy as np
import scipy.sparse as sp
import random
import argparse


random.seed(226)


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


eigs_np = eigenvalues_np(matrix.toarray(), symmetric=True)
eigs_sp = eigenvalues_sp(matrix, symmetric=True)
eigs_cp = eigenvalues_cp(cp_symm_matrix)


index_np = np.argmax(np.abs(eigs_np))
index_sp = np.argmax(np.abs(eigs_sp))
index_cp = np.argmax(np.abs(eigs_cp))


biggest_eigenvalue_np = eigs_np[index_np]
biggest_eigenvalue_sp = eigs_sp[index_sp]
biggest_eigenvalue_cp = eigs_cp[index_cp]

biggest_eigenvalue_pm = power_method(matrix)
# biggest_eigenvalue_pm_numba = power_method_numba(matrix.toarray())
biggest_eigenvalue_cp = power_method_cp(cp_symm_matrix)

_, __ = QR(A)
_, __ = QR_cp(A)
