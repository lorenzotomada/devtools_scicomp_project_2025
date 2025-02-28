import numpy as np
import scipy.sparse as sp
from line_profiler import profile
from numpy.linalg import eig, eigh
from pyclassify.utils import check_A_square_matrix, power_method_numba_helper


@profile
def eigenvalues_np(A, symmetric=True):
    check_A_square_matrix(A)
    eigenvalues, _ = eigh(A) if symmetric else eig(A)
    return eigenvalues


@profile
def eigenvalues_sp(A, symmetric=True):
    check_A_square_matrix(A)
    eigenvalues, _ = (
        sp.linalg.eigsh(A, k=A.shape[0] - 1)
        if symmetric
        else sp.linalg.eigs(A, k=A.shape[0] - 1)
    )
    return eigenvalues


@profile
def power_method(A, max_iter=500, tol=1e-4, x=None):
    check_A_square_matrix(A)
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


@profile
def power_method_numba(A):
    return power_method_numba_helper(A)
