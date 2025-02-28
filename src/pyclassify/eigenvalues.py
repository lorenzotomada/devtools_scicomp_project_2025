import numpy as np
import scipy.sparse as sp
from line_profiler import profile
from numpy.linalg import eig, eigh
from pyclassify.utils import check_A_square_matrix, power_method_numba_helper


@profile
def eigenvalues_np(A, symmetric=True):
    """
    Compute the eigenvalues of a square matrix using NumPy's `eig` or `eigh` function.

    This function checks if the input matrix is square (and is actually a matrix) using 'check_A_square_matrix', and then computes its eigenvalues.
    If the matrix is symmetric, it uses `eigh` (which is more efficient for symmetric matrices).
    Otherwise, it uses `eig`.

    Args:
        A (np.ndarray): A square matrix whose eigenvalues are to be computed.
        symmetric (bool, optional): If True, assumes the matrix is symmetric and uses `eigh` for
                                     faster computation. If False, uses `eig` for general matrices
                                     (default is True).

    Returns:
        np.ndarray: An array containing the eigenvalues of the matrix `A`.

    Raises:
        TypeError: If the input is not a NumPy array or a SciPy sparse matrix.
        ValueError: If number of rows != number of columns.
    """
    check_A_square_matrix(A)
    eigenvalues, _ = eigh(A) if symmetric else eig(A)
    return eigenvalues


@profile
def eigenvalues_sp(A, symmetric=True):
    """
    Compute the eigenvalues of a sparse matrix using SciPy's `eigsh` or `eigs` function.

    This function checks if the input matrix is square, then computes its eigenvalues using
    SciPy's sparse linear algebra solvers. For symmetric matrices, it uses `eigsh` for
    more efficient computation. For non-symmetric matrices, it uses `eigs`.

    Args:
        A (sp.spmatrix): A square sparse matrix whose eigenvalues are to be computed.
        symmetric (bool, optional): If True, assumes the matrix is symmetric and uses `eigsh`.
                                     If False, uses `eigs` for general matrices (default is True).

    Returns:
        np.ndarray: An array containing the eigenvalues of the sparse matrix `A`.

    Raises:
        TypeError: If the input is not a NumPy array or a SciPy sparse matrix.
        ValueError: If number of rows != number of columns.
    """
    check_A_square_matrix(A)
    eigenvalues, _ = (
        sp.linalg.eigsh(A, k=A.shape[0] - 1)
        if symmetric
        else sp.linalg.eigs(A, k=A.shape[0] - 1)
    )
    return eigenvalues


@profile
def power_method(A, max_iter=500, tol=1e-4, x=None):
    """
    Compute the dominant eigenvalue of a square matrix using the power method.

    Args:
        A (np.ndarray or sp.spmatrix): A square matrix whose dominant eigenvalue is to be computed.
        max_iter (int, optional): Maximum number of iterations to perform (default is 500).
        tol (float, optional): Tolerance for convergence based on the relative change between iterations
                               (default is 1e-4).
        x (np.ndarray, optional): Initial guess for the eigenvector. If None, a random vector is generated.

    Returns:
        float: The approximated dominant eigenvalue of the matrix `A`, computed as the Rayleigh quotient x @ A @ x.

    Raises:
        TypeError: If the input is not a NumPy array or a SciPy sparse matrix.
        ValueError: If number of rows != number of columns.
    """
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
    """
    Compute the dominant eigenvalue of a matrix using the power method, with Numba optimization.

    This function wraps the `power_method` function and applies Numba's Just-In-Time (JIT) compilation
    to optimize the performance of the power method for large matrices. The function leverages the
    `power_method_numba_helper` function for the actual computation. The reason for that is the following: profiling
    directly a function decorated with @numba.jit does not actually keep track of the calls and execution time due to
    numba technicalities. Therefore, the helper function is profiled instead.
    Remark that numba does not support scipy sparse matrices, so the input matrix must be a NumPy array.

    Args:
        A (np.ndarray): A square matrix whose dominant eigenvalue is to be computed.

    Returns:
        float: The approximated dominant eigenvalue of the matrix `A`.
    """
    return power_method_numba_helper(A)
