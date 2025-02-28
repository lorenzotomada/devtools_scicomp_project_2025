import numpy as np
import scipy.sparse as sp
import numba
import os
import yaml
from line_profiler import profile

# from numba.pycc import CC


def check_A_square_matrix(A):
    if not isinstance(A, (np.ndarray, sp.spmatrix)):
        raise TypeError("Input matrix must be a NumPy array or a SciPy sparse matrix!")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square!")


@profile
def make_symmetric(A):
    check_A_square_matrix(A)
    A_sym = (A + A.T) / 2
    return A_sym


@numba.njit(nogil=True, parallel=True)
def power_method_numba_helper(A, max_iter=500, tol=1e-4, x=None):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square!")  # explain why re-written
    if x is None:
        x = np.random.rand(A.shape[0])
    x /= np.linalg.norm(x)
    x_old = x

    iteration = 0
    update_norm = tol + 1

    while iteration < max_iter and update_norm > tol:
        x = A @ x
        x /= np.linalg.norm(x)
        update_norm = np.linalg.norm(x - x_old) / np.linalg.norm(x_old)
        x_old = x.copy()
        iteration += 1

    return x @ A @ x


def read_config(file: str) -> dict:
    """
    To read the desired configuration file, passed in input as a string
    Input:
        file: str (representing the location of file)
    Returns:
        dict (containing the configuration parameters needed)
    """
    filepath = os.path.abspath(f"{file}.yaml")
    with open(filepath, "r") as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs
