import numpy as np
import scipy.sparse as sp
import scipy.linalg as spla
from line_profiler import profile
from numpy.linalg import eig, eigh
from pyclassify.utils import (
    check_square_matrix,
    check_symm_square,
    power_method_numba_helper,
)
import cupy as cp

# cp.cuda.Device(0)
import cupy.linalg as cpla
from cupyx.scipy.sparse.linalg import eigsh as eigsh_cp


@profile
def eigenvalues_np(A, symmetric=True):
    """
    Compute the eigenvalues of a square matrix using NumPy's `eig` or `eigh` function.

    This function checks if the input matrix is square (and is actually a matrix) using 'check_square_matrix', and then computes its eigenvalues.
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
    check_square_matrix(A)
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
    check_square_matrix(A)
    eigenvalues, _ = (
        sp.linalg.eigsh(A, k=A.shape[0] - 1)
        if symmetric
        else sp.linalg.eigs(A, k=A.shape[0] - 1)
    )
    return eigenvalues


@profile
def eigenvalues_cp(A):
    """
    Compute the eigenvalues of a sparse matrix using CuPy's `eigsh` function.

    This function checks if the input matrix is square and symmetric, then computes its eigenvalues using
    CuPy's sparse linear algebra solvers. It uses `eigsh` for more efficient computation.
    Remark that the eigsh function in this case does not allow to compute *all* the eigenvalues, but only a number
    $m<n$, so here just a reduced portion is computed (starting form the ones which are greater in magnitude).

    Args:
        A (cpsp.spmatrix): A square sparse matrix whose eigenvalues are to be computed.

    Returns:
        np.ndarray: An array containing the eigenvalues of the sparse matrix `A`.

    Raises:
        TypeError: If the input is not a CuPy sparse symmetric matrix.
        ValueError: If number of rows != number of columns.
    """
    check_symm_square(A)
    k = 5 if A.shape[0] > 5 else A.shape[0] - 2
    eigenvalues, _ = eigsh_cp(A, k=k)
    return eigenvalues


@profile
def power_method(A, max_iter=500, tol=1e-4, x=None):
    """
    Compute the dominant eigenvalue of a square matrix using the power method.

    Args:
        A (np.ndarray or sp.spmatrix or cpsp.spmatrix): A square matrix whose dominant eigenvalue is to be computed.
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
    check_square_matrix(A)
    if x is None:
        x = np.random.rand(A.shape[0])
    x /= spla.norm(x)
    x_old = x

    iteration = 0
    update_norm = tol + 1

    while iteration < max_iter and update_norm > tol:
        x = A @ x
        x /= spla.norm(x)
        update_norm = spla.norm(x - x_old) / spla.norm(x_old)
        x_old = x.copy()
        iteration += 1

    return x @ A @ x


@profile
def power_method_numba(A, max_iter=500, tol=1e-4, x=None):
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
    return power_method_numba_helper(A, max_iter, tol, x)


@profile
def power_method_cp(A, max_iter=500, tol=1e-4, x=None):
    """
    Compute the dominant eigenvalue of a square matrix using the power method.

    Args:
        A (cp.spmatrix): A square matrix whose dominant eigenvalue is to be computed.
        max_iter (int, optional): Maximum number of iterations to perform (default is 500).
        tol (float, optional): Tolerance for convergence based on the relative change between iterations
                               (default is 1e-4).
        x (cp.ndarray, optional): Initial guess for the eigenvector. If None, a random vector is generated.

    Returns:
        float: The approximated dominant eigenvalue of the matrix `A`, computed as the Rayleigh quotient x @ A @ x.

    Raises:
        TypeError: If the input is not a NumPy array or a SciPy sparse matrix.
        ValueError: If number of rows != number of columns.
    """
    check_square_matrix(A)
    if x is None:
        x = cp.random.rand(A.shape[0])
    x /= cpla.norm(x)
    x_old = x

    iteration = 0
    update_norm = tol + 1

    while iteration < max_iter and update_norm > tol:
        x = A @ x
        x /= cpla.norm(x)
        update_norm = cpla.norm(x - x_old) / cpla.norm(x_old)
        x_old = x.copy()
        iteration += 1

    return x @ A @ x


def Lanczos_PRO(A, q, m=None, toll=np.sqrt(np.finfo(float).eps)):
    """
    Perform the Lanczos algorithm for symmetric matrices.

    This function computes an orthogonal matrix Q and tridiagonal matrix T such that A ≈ Q * T * Q.T,
    where A is a symmetric matrix. The algorithm is useful for finding a few eigenvalues and eigenvectors
    of large symmetric matrices.

    Args:
        A (np.ndarray): A symmetric square matrix of size n x n.
        q (np.ndarray): Initial vector of size n.
        m (int, optional): Number of eigenvalues to compute. Must be less than or equal to n.
                           If None, defaults to the size of A.
        toll (float, optional): Tolerance for orthogonality checks (default is sqrt(machine epsilon)).

    Returns:
        tuple: A tuple (Q, alpha, beta) where:
            - Q (np.ndarray): Orthogonal matrix of size n x m.
            - alpha (np.ndarray): Vector of size m containing the diagonal elements of the tridiagonal matrix.
            - beta (np.ndarray): Vector of size m-1 containing the off-diagonal elements of the tridiagonal matrix.

    Raises:
        ValueError: If the input matrix A is not square or if m is greater than the size of A.
    """
    if m == None:
        m = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix A must be square.")

    if A.shape[0] != q.shape[0]:
        raise ValueError("Input vector q must have the same size as the matrix A.")

    q = q / np.linalg.norm(q)
    Q = np.array([q])
    r = A @ q
    alpha = []
    beta = []
    alpha.append(q @ r)
    r = r - alpha[0] * q
    beta.append(np.linalg.norm(r))
    count = 0
    for j in range(1, m):
        q = r / beta[j - 1]
        for q_basis in Q[:-1]:
            if np.abs(q @ q_basis) > toll:
                for q_bbasis in Q[:-1]:
                    q = q - (q @ q_bbasis) * q_bbasis
                    count += 1
                break
        q = q / np.linalg.norm(q)
        Q = np.vstack((Q, q))
        r = A @ q - beta[j - 1] * Q[j - 1]
        alpha.append(q @ r)
        r = r - alpha[j] * q
        beta.append(np.linalg.norm(r))

        if np.abs(beta[j]) < 1e-15:
            return Q, alpha, beta[:-1]
    return Q, alpha, beta[:-1]


def QR_method(A_copy, tol=1e-10, max_iter=100):
    """
    Compute the eigenvalues of a tridiagonal matrix using the QR algorithm.

    This function uses the QR decomposition method to iteratively compute the eigenvalues of a given tridiagonal matrix.
    The QR algorithm is an iterative method that computes the eigenvalues of a matrix by decomposing it into a product
    of an orthogonal matrix Q and an upper triangular matrix R, and then updating the matrix as the product of R and Q.

    Args:
        A_copy (np.ndarray): Atridiagonal matrix whose eigenvalues are to be computed.
        tol (float, optional): Tolerance for convergence based on the off-diagonal elements (default is 1e-10).
        max_iter (int, optional): Maximum number of iterations to perform (default is 100).

    Returns:
        tuple: A tuple (eigenvalues, Q) where:
            - eigenvalues (np.ndarray): An array containing the eigenvalues of the matrix `A_copy`.
            - Q (np.ndarray): The orthogonal matrix Q from the final QR decomposition.

    Raises:
        ValueError: If the input matrix A_copy is not square.
    """

    A = A_copy.copy()
    A = np.array(A)
    iter = 0
    Q = np.array([])

    while np.linalg.norm(np.diag(A, -1), np.inf) > tol and iter < max_iter:
        Matrix_trigonometry = np.array([])
        for i in range(A.shape[0] - 1):
            c = A[i, i] / np.sqrt(A[i, i] ** 2 + A[i + 1, i] ** 2)
            s = -A[i + 1, i] / np.sqrt(A[i, i] ** 2 + A[i + 1, i] ** 2)
            Matrix_trigonometry = (
                np.vstack((Matrix_trigonometry, [c, s]))
                if Matrix_trigonometry.size
                else np.array([[c, s]])
            )

            R = np.array([[c, -s], [s, c]])
            A[i : i + 2, i : i + 3] = R @ A[i : i + 2, i : i + 3]
            A[i + 1, i] = 0
        Q = np.eye(A.shape[0])
        i = 0
        Q[0:2, 0:2] = np.array(
            [
                [Matrix_trigonometry[i, 0], Matrix_trigonometry[i, 1]],
                [-Matrix_trigonometry[i, 1], Matrix_trigonometry[i, 0]],
            ]
        )
        for i in range(1, A.shape[0] - 1):
            R = np.eye(A.shape[0])
            R[i : i + 2, i : i + 2] = np.array(
                [
                    [Matrix_trigonometry[i, 0], Matrix_trigonometry[i, 1]],
                    [-Matrix_trigonometry[i, 1], Matrix_trigonometry[i, 0]],
                ]
            )
            Q = Q @ R
        A = A @ Q
        iter += 1

    return np.diag(A), Q
