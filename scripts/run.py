from pyclassify import (
    eigenvalues_np,
    eigenvalues_sp,
    power_method,
    power_method_numba,
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
# also tol and max iter


matrix = sp.random(dim, dim, density=density, format="csr")
matrix = make_symmetric(matrix)

eigs_np = eigenvalues_np(matrix.toarray(), symmetric=True)
eigs_sp = eigenvalues_sp(matrix, symmetric=True)

index_np = np.argmax(np.abs(eigs_np))
index_sp = np.argmax(np.abs(eigs_sp))

biggest_eigenvalue_np = eigs_np[index_np]
biggest_eigenvalue_sp = eigs_sp[index_sp]

biggest_eigenvalue_pm = power_method(matrix)
biggest_eigenvalue_pm_numba = power_method_numba(matrix.toarray())
