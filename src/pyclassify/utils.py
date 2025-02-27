import numpy as np
import scipy.sparse as sp
from numpy.linalg import eig, eigh



def make_symmetric(A):
    if not sp.isspmatrix(A):
        raise TypeError("Input must be a scipy sparse matrix")
    
    A_sym = (A + A.T) / 2
    return A_sym



def eigenvalues_np(A, symmetric=True):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square!")
    if not isinstance(A, (np.ndarray, sp.spmatrix)):
        raise TypeError("Input matrix must be a NumPy array or a SciPy sparse matrix!")
    
    eigenvalues, _ = eigh(A) if symmetric else eig(A)
    return eigenvalues



def eigenvalues_sp(A, symmetric=True):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square!")
    if not isinstance(A, (np.ndarray, sp.spmatrix)):
        raise TypeError("Input matrix must be a NumPy array or a SciPy sparse matrix!")

    eigenvalues, _ = sp.linalg.eigsh(A, k=A.shape[0] - 1) if symmetric else sp.linalg.eigs(A, k=A.shape[0] - 1)
    return eigenvalues



def power_method(A, max_iter=1000, tol=1e-4):
    assert A.shape[0] == A.shape[1], "Matrix must be square"

    x = np.random.rand(A.shape[0])
    x /= np.linalg.norm(x)
    x_old = x

    iteration = 0
    update_norm = tol + 1

    while iteration < max_iter and update_norm > tol:
        x = A @ x
        x /= np.linalg.norm(x)
        update_norm = np.linalg.norm(x - x_old)/np.linalg.norm(x_old)
        x_old = x.copy()
        iteration += 1

    return x @ A @ x