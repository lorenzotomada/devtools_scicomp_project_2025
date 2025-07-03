import numpy as np
import scipy.sparse as sp
import scipy.linalg as spla
from numpy.linalg import eig, eigh
from numba import jit, prange
from .cxx_utils import QR_algorithm, Eigen_value_calculator
from pyclassify.utils import (
    check_square_matrix,
    check_symm_square,
    max_iteration_warning,
)

# from parallel_tridiag_eigen import parallel_eigen


def eigenvalues_np(A, symmetric=True):
    """
    Wrapper for the np eigenvalue solver. This function is only used in tests for better readability.
    Compute the eigenvalues of a square matrix using NumPy's `eig` or `eigh` function.

    This function checks if the input matrix is square (and is actually a matrix) using
    'check_square_matrix', and then computes its eigenvalues.

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
        TypeError: If the input is not a NumPy array or a SciPy/CuPy sparse matrix.
        ValueError: If number of rows != number of columns.
    """
    check_square_matrix(A)
    eigenvalues, _ = eigh(A) if symmetric else eig(A)
    return eigenvalues


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
        TypeError: If the input is not a NumPy array or a SciPy/CuPy sparse matrix.
        ValueError: If number of rows != number of columns.
    """
    check_square_matrix(A)
    eigenvalues, _ = (
        sp.linalg.eigsh(A, k=A.shape[0] - 1)
        if symmetric
        else sp.linalg.eigs(A, k=A.shape[0] - 1)
    )
    return eigenvalues


def power_method(A, max_iter=500, tol=1e-7, x=None):
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
        TypeError: If the input is not a NumPy array or a SciPy/CuPy sparse matrix.
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
        if iteration >= max_iter:
            max_iteration_warning()
    return x @ A @ x


@jit(nopython=True, nogil=True, parallel=True)
def power_method_numba(A, max_iter=500, tol=1e-7, x=None):
    """
    Compute the dominant eigenvalue of a matrix using the power method, with Numba optimization.

    This function and applies Numba's Just-In-Time (JIT) compilation to optimize the performance of the
    power method for large matrices.

    Remark that numba does not support scipy sparse matrices, so the input matrix must be a NumPy array.
    he function is optimized with Numba using the 'njit' decorator with nogil and parallel options.
    We have re-written the code due to the fact that using numba we cannot use the helper function check_square_matrix.

    Args:
        A (np.ndarray): A square matrix.
        max_iter (int, optional): Maximum number of iterations to perform (default is 500).
        tol (float, optional): Tolerance for convergence based on the relative change between
                               iterations (default is 1e-4).
        x (np.ndarray, optional): Initial guess for the eigenvector. If None, a random vector is generated.

    Returns:
        float: The approximated dominant eigenvalue of the matrix A.

    Raises:
        ValueError: If the input matrix A is not square. The check is not done using 'check_square_matrix'
                    because of numba technicalities.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            "Matrix must be square!"
        )  # not possible to use the helper function due to the fact that we are using JIT-compilation
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
        # if iteration >= max_iter:
        #    max_iteration_warning()
    return x @ A @ x


@jit(nopython=True)
def Lanczos_PRO(A, q, m=None, tol=np.sqrt(np.finfo(float).eps)):
    r"""
    Perform the Lanczos algorithm for symmetric matrices.

    This function computes an orthogonal matrix Q and tridiagonal matrix T such that
    .. math:: `A \approx Q T Q^T,`
    where A is a symmetric matrix. The algorithm is useful for finding a few eigenvalues and eigenvectors
    of large symmetric matrices.

    Args:
        A (np.ndarray): A symmetric square matrix of size n x n.
        q (np.ndarray, optional): Initial vector of size n. Default value is None (a random one is created).
        m (int, optional): Number of eigenvalues to compute. Must be less than or equal to n.
                If None, defaults to the size of A.
        tol (float, optional): Tolerance for orthogonality checks (default is sqrt(machine epsilon)).

    Returns:
        tuple: A tuple (Q, alpha, beta) where:
            - Q (np.ndarray): Orthogonal matrix of size n x m.
            - alpha (np.ndarray): Vector of size m containing the diagonal elements of the tridiagonal matrix.
            - beta (np.ndarray): Vector of size m-1 containing the off-diagonal elements of the tridiagonal matrix.

    Raises:
        TypeError: If the input is not a NumPy array or SciPy/CuPy sparse matrix.
        ValueError: If number of rows != number of columns or the matrix is not symmetric or it m is
                    greater than the size of A.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            "Matrix must be square!"
        )  # not possible to use the helper function due to the fact that we are using JIT-compilation

    if m == None:
        m = A.shape[0]

    if m > A.shape[0]:
        raise ValueError("The value of m cannot be greater than the size of A!")

    q = q / np.linalg.norm(q)
    Q = np.zeros((m, A.shape[0]))
    Q[0] = q
    r = A @ q
    alpha = []
    beta = []
    alpha.append(q @ r)
    r = r - alpha[0] * q
    beta.append(np.linalg.norm(r))

    for j in range(1, m):
        q = r / beta[j - 1]
        if np.any(np.abs(q @ Q[: j - 1].T) > tol):
            partial = np.zeros((j, len(q)), dtype=np.float64)
            for i in prange(j):
                h = 0.0
                # Compute the dot product: h = q dot Q[i]

                h = q @ Q[i]
                # Store the contribution: h * Q[i] into the ith row

                partial[i] = h * Q[i]

            # Reduce the contributions (summing the partial projections)
            q = q - np.sum(partial, axis=0)

        q = q / np.linalg.norm(q)
        Q[j] = q
        r = A @ q - beta[j - 1] * Q[j - 1]
        alpha.append(q @ r)
        r = r - alpha[j] * q
        beta.append(np.linalg.norm(r))

        if np.abs(beta[j]) < 1e-15:
            alpha = np.array(alpha)
            beta = np.array(beta)
            return Q, alpha, beta[:-1]

    alpha = np.array(alpha)
    beta = np.array(beta)
    return Q, alpha, beta[:-1]


class EigenSolver:
    """
    Class solving the eigenvalue problem for a given symmetric matrix.

    Two different building blocks are present: Lanczos_PRO (used to transform the matrix to
    a tridiagonal one), and another function written in C++. The latter can be either
    Eigen_value_calculator (if the user is only interested in the computation of the eigenvalues)
    or QR_algorithm, if eigenvectors are needed as well.

    We refer the interested reader to their implementation in C++ for further details.
    """

    def __init__(self, A: np.ndarray, max_iter=5000, tol=1e-8, tol_deflation=1e-12):
        """
        Class constructor. Takes as input a matrix A (supposed to be symmetric), the number
        of iterations that are allowed and a tolerance.

        Args:
            A (np.ndarray): A square matrix whose eigenvalues and (possibly) eigenvectors are to
            be computed
            max_iter (int, optional): Maximum number of iterations to perform (default is 5000)
            tol (float, optional): Tolerance for convergence (default is 1e-8).

        Raises:
        TypeError: If the input is not a NumPy array or SciPy/CuPy sparse matrix.
        ValueError: If number of rows != number of columns or the matrix is not symmetric.
        """
        check_symm_square(A)

        self.A = A
        self.tol = tol
        self.max_iter = max_iter
        self.diag = None
        self.off_diag = None
        self.Q = None
        self.tol_deflation = tol_deflation

    def Lanczos_PRO(self, A=None, q=None, m=None, tol=np.sqrt(np.finfo(float).eps)):
        r"""
        Perform the Lanczos algorithm for symmetric matrices.

        This function computes an orthogonal matrix Q and tridiagonal matrix T such that
        .. math:: `A \approx Q T Q^T,`
        where A is a symmetric matrix. The algorithm is useful for finding a few eigenvalues and eigenvectors
        of large symmetric matrices.

        Args:
            A (np.ndarray): A symmetric square matrix of size n x n.
            q (np.ndarray, optional): Initial vector of size n. Default value is None (a random one is created).
            m (int, optional): Number of eigenvalues to compute. Must be less than or equal to n.
                    If None, defaults to the size of A.
            tol (float, optional): Tolerance for orthogonality checks (default is sqrt(machine epsilon)).

        Returns:
            tuple: A tuple (Q, alpha, beta) where:
                - Q (np.ndarray): Orthogonal matrix of size n x m.
                - alpha (np.ndarray): Vector of size m containing the diagonal elements of the tridiagonal matrix.
                - beta (np.ndarray): Vector of size m-1 containing the off-diagonal elements of the tridiagonal matrix.

        Raises:
            TypeError: If the input is not a NumPy array or SciPy/CuPy sparse matrix.
            ValueError: If number of rows != number of columns or the matrix is not symmetric or it m is
                        greater than the size of A.
        """

        if A is None:
            A = self.A

        if q is None:
            q = np.random.rand(A.shape[0])
            if q[0] == 0:
                q[0] += 1

        else:
            check_square_matrix(A)

        if m == None:
            m = A.shape[0]

        if m > A.shape[0]:
            raise ValueError("The value of m cannot be greater than the size of A!")

        q = q / np.linalg.norm(q)
        Q = np.zeros((m, A.shape[0]))
        Q[0] = q
        r = A @ q
        alpha = []
        beta = []
        alpha.append(q @ r)
        r = r - alpha[0] * q
        beta.append(np.linalg.norm(r))

        for j in range(1, m):
            q = r / beta[j - 1]
            if np.any(np.abs(q @ Q[: j - 1].T) > tol):
                partial = np.zeros((j, len(q)), dtype=np.float64)
                for i in prange(j):
                    h = 0.0
                    # Compute the dot product: h = q dot Q[i]

                    h = q @ Q[i]
                    # Store the contribution: h * Q[i] into the ith row

                    partial[i] = h * Q[i]

                # Reduce the contributions (summing the partial projections)
                q = q - np.sum(partial, axis=0)

            q = q / np.linalg.norm(q)
            Q[j] = q
            r = A @ q - beta[j - 1] * Q[j - 1]
            alpha.append(q @ r)
            r = r - alpha[j] * q
            beta.append(np.linalg.norm(r))

            if np.abs(beta[j]) < 1e-15:
                alpha = np.array(alpha)
                beta = np.array(beta)
                self.diag = alpha
                self.off_diag = beta[:-1]
                self.Q = Q
                return Q, alpha, beta[:-1]

        alpha = np.array(alpha)
        beta = np.array(beta)
        self.diag = alpha
        self.off_diag = beta[:-1]
        self.Q = np.array(Q)
        return Q, alpha, beta[:-1]

    @property
    def initial_guess(self):
        q = np.random.rand(self.A.shape[0])
        if q[0] == 0:
            q[0] += 1
        return q

    def compute_eigenval(self, diag=None, off_diag=None):
        """
        Compute (only) the eigenvalues of a symmetric triangular matrix, passed as argument in the form of diagonal and
        off-diagonal.

        This function relies on the Eigen_value_calculator function, written in C++.
        Args:
            diag (np.ndarray, optional): Diagonal of the matrix. Default value is None. If no value is passed, the one used
                                        is the one resulting from the Lanczos decomposition of the matrix passed to the
                                        constructor.
            off_diag (np.ndarray, optional): Off-iagonal of the matrix. Default value is None. If no value is passed, the one
                                        used is the one resulting from the Lanczos decomposition of the matrix passed to the
                                        constructor.

        Returns:
            np.array: an np.array containing the eigenvalues of the matrix.

        Raises:
            ValueError: If there is a mismatch between the diagonal and off diagonal size.
        """
        if diag is None and off_diag is None:
            if self.diag is None:
                self.Q, self.diag, self.off_diag = Lanczos_PRO(
                    A=self.A, q=self.initial_guess, tol=self.tol
                )
            diag = self.diag
            off_diag = self.off_diag
        if len(diag) != (len(off_diag) + 1):
            raise ValueError("Mismatch between diagonal and off diagonal size")

        return np.array(Eigen_value_calculator(diag, off_diag, self.tol, self.max_iter))

    def eig(self, diag=None, off_diag=None):
        """
        Compute the eigenvalues and eigenvectors of a symmetric triangular matrix, passed as argument in the form of
        diagonal and off-diagonal.

        This function relies on the QR_algorithm function, written in C++.
        Args:
            diag (np.ndarray, optional): Diagonal of the matrix. Default value is None. If no value is passed, the one used
                                        is the one resulting from the Lanczos decomposition of the matrix passed to the
                                        constructor.
            off_diag (np.ndarray, optional): Off-iagonal of the matrix. Default value is None. If no value is passed, the one
                                        used is the one resulting from the Lanczos decomposition of the matrix passed to the
                                        constructor.

        Returns:
            tuple: a tuple containing two arrays, the eigenvalues and the eigenvectors.

        Raises:
            ValueError: If there is a mismatch between the diagonal and off diagonal size.
        """
        if diag is None and off_diag is None:
            if self.diag is None:
                self.Q, self.diag, self.off_diag = Lanczos_PRO(
                    A=self.A, q=self.initial_guess, tol=self.tol
                )
            diag = self.diag
            off_diag = self.off_diag
        if len(diag) != (len(off_diag) + 1):
            raise ValueError("Mismatch  between diagonal and off diagonal size")

        eig, Q_triangular = QR_algorithm(diag, off_diag, self.tol, self.max_iter)
        Q_triangular = np.array(Q_triangular)
        return np.array(eig), Q_triangular @ self.Q.T


# import cupy as cp
# import cupy.linalg as cpla
# from cupyx.scipy.sparse.linalg import eigsh as eigsh_cp
#
#
#
# def eigenvalues_cp(A):
#    """
#    Compute the eigenvalues of a sparse matrix using CuPy's `eigsh` function.
#
#    This function checks if the input matrix is square and symmetric, then computes its eigenvalues using
#    CuPy's sparse linear algebra solvers. It uses `eigsh` for more efficient computation.
#
#    Remark that the eigsh function in this case does not allow to compute *all* the eigenvalues, but only a number
#    m<n, so here just a reduced portion is computed (starting form the ones which are greater in magnitude).
#
#    Args:
#        A (cpsp.spmatrix): A square sparse matrix whose eigenvalues are to be computed.
#
#    Returns:
#        np.ndarray: An array containing the eigenvalues of the sparse matrix `A`.
#
#    Raises:
#        TypeError: If the input is not a NumPy array or Scipy/CuPy sparse symmetric matrix.
#        ValueError: If number of rows != number of columns.
#    """
#    check_symm_square(A)
#    k = 5 if A.shape[0] > 5 else A.shape[0] - 2
#    eigenvalues, _ = eigsh_cp(A, k=k)
#    return eigenvalues
#
# def power_method_cp(A, max_iter=500, tol=1e-4, x=None):
#    """
#    Compute the dominant eigenvalue of a square matrix using the power method.
#    Implemented using cupy.
#
#    Args:
#        A (cp.spmatrix): A square matrix whose dominant eigenvalue is to be computed.
#        max_iter (int, optional): Maximum number of iterations to perform (default is 500).
#        tol (float, optional): Tolerance for convergence based on the relative change between iterations
#                               (default is 1e-4).
#        x (cp.ndarray, optional): Initial guess for the eigenvector. If None, a random vector is generated.
#
#    Returns:
#        float: The approximated dominant eigenvalue of the matrix `A`, computed as the Rayleigh quotient x @ A @ x.
#
#    Raises:
#        TypeError: If the input is not a NumPy array or a SciPy sparse matrix.
#        ValueError: If number of rows != number of columns.
#    """
#    check_square_matrix(A)
#    if x is None:
#        x = cp.random.rand(A.shape[0])
#    x /= cpla.norm(x)
#    x_old = x
#
#    iteration = 0
#    update_norm = tol + 1
#
#    while iteration < max_iter and update_norm > tol:
#        x = A @ x
#        x /= cpla.norm(x)
#        update_norm = cpla.norm(x - x_old) / cpla.norm(x_old)
#        x_old = x.copy()
#        iteration += 1
#        if iteration >= max_iter:
#            max_iteration_warning()
#
#    return x @ A @ x
#
#
# def Lanczos_PRO_cp(A, q=None, m=None, tol=1e-8):
#    r"""
#    Perform the Lanczos algorithm for symmetric matrices.
#
#    This function computes an orthogonal matrix Q and tridiagonal matrix T such that A is approximately
#    equal to Q * T * Q.T, where A is a symmetric matrix. The algorithm is useful for finding a few
#    eigenvalues and eigenvectors of large symmetric matrices.
#    Implemented using cupy.
#
#    Args:
#        A (cp.ndarray or cpsp.spmatrix): A symmetric square matrix of size n x n.
#        q (cp.ndarray): Initial vector of size n.
#        m (int, optional): Number of eigenvalues to compute. Must be less than or equal to n.
#                           If None, defaults to the size of A.
#        tol (float, optional): Tolerance for orthogonality checks (default is sqrt(machine epsilon)).
#
#    Returns:
#        tuple: A tuple (Q, alpha, beta) where:
#            - Q (cp.ndarray): Orthogonal matrix of size n x m.
#            - alpha (cp.ndarray): Vector of size m containing the diagonal elements of the tridiagonal matrix.
#            - beta (cp.ndarray): Vector of size m-1 containing the off-diagonal elements of the tridiagonal matrix.
#
#    Raises:
#        ValueError: If the input matrix A is not square or if m is greater than the size of A.
#    """
#    if q is None:
#        q = cp.random.rand(A.shape[0])
#        if q[0] == 0:
#            q[0] += 1
#
#    if m == None:
#        m = A.shape[0]
#
#    check_symm_square(A)
#
#    if A.shape[0] != q.shape[0]:
#        raise ValueError("Input vector q must have the same size as the matrix A.")
#
#    q = q / cp.linalg.norm(q)
#    # Q=np.array([q])
#    Q = cp.zeros((m, A.shape[0]))
#    Q[0] = q
#    r = A @ q
#    alpha = []
#    beta = []
#    alpha.append(q @ r)
#    r = r - alpha[0] * q
#    beta.append(cp.linalg.norm(r))
#
#    for j in range(1, m):
#        q = r / beta[j - 1]
#        if cp.any(cp.abs(q @ Q[: j - 1].T) > tol):
#            for q_bbasis in Q[: j - 1]:
#                q = q - (q @ q_bbasis) * q_bbasis
#
#        q = q / cp.linalg.norm(q)
#        Q[j] = q
#        r = A @ q - beta[j - 1] * Q[j - 1]
#        alpha.append(q @ r)
#        r = r - alpha[j] * q
#        beta.append(cp.linalg.norm(r))
#
#        if cp.abs(beta[j]) < 1e-15:
#            return Q, cp.array(alpha), cp.array(beta[:-1])
#    return Q, cp.array(alpha), cp.array(beta[:-1])
#
#
# def QR_method_cp(diag, off_diag, tol=1e-8, max_iter=1000):
#    """
#    Compute the eigenvalues of a tridiagonal matrix using the QR algorithm.
#
#    This function uses the QR decomposition method to iteratively compute the eigenvalues of a given tridiagonal matrix.
#    The QR algorithm is an iterative method that computes the eigenvalues of a matrix by decomposing it into a product
#    of an orthogonal matrix Q and an upper triangular matrix R, and then updating the matrix as the product of R and Q.
#    Implemented using cupy.
#
#    Args:
#        diag (cp.ndarray): Diagonal elements of the tridiagonal matrix.
#        off_diag (cp.ndarray): Off-diagonal elements of the tridiagonal matrix.
#        tol (float, optional): Tolerance for convergence based on the off-diagonal elements (default is 1e-10).
#        max_iter (int, optional): Maximum number of iterations to perform (default is 100).
#
#    Returns:
#        tuple: A tuple (eigenvalues, Q) where:
#            - eigenvalues (cp.ndarray): An array containing the eigenvalues of the matrix.
#            - Q (cp.ndarray): The orthogonal matrix Q from the final QR decomposition.
#
#    Raises:
#        ValueError: If the input matrix is not square.
#    """
#    n = diag.shape[0]
#    Q = cp.eye(n)
#
#    Matrix_trigonometric = cp.zeros((n - 1, 2))
#
#    iter = 0
#    # eigenvalues_old = np.array(diag)
#
#    r, c, s = 0, 0, 0
#    d, mu = 0, 0  # mu: Wilkinson shift
#    a_m, b_m_1 = 0, 0
#    x, y = 0, 0
#    m = n - 1
#    tol_equivalence = 1e-10
#    w, z = 0, 0
#
#    while iter < max_iter and m > 0:
#        # prefetching most used value to avoid call overhead
#        a_m = diag[m]
#        b_m_1 = off_diag[m - 1]
#        d = (diag[m - 1] - a_m) * 0.5
#
#        if cp.abs(d) < tol_equivalence:
#            mu = diag[m] - cp.abs(b_m_1)
#        else:
#            mu = a_m - b_m_1 * b_m_1 / (
#                d * (1 + cp.sqrt(d * d + b_m_1 * b_m_1) / cp.abs(d))
#            )
#
#        x = diag[0] - mu
#        y = off_diag[0]
#
#        for i in range(m):
#            if m > 1:
#                r = np.sqrt(x * x + y * y)
#                c = x / r
#                s = -y / r
#                Matrix_trigonometric[i][0] = c
#                Matrix_trigonometric[i][1] = s
#
#                w = c * x - s * y
#                d = diag[i] - diag[i + 1]
#                z = (2 * c * off_diag[i] + d * s) * s
#                diag[i] -= z
#                diag[i + 1] += z
#                off_diag[i] = d * c * s + (c * c - s * s) * off_diag[i]
#                x = off_diag[i]
#                if i > 0:
#                    off_diag[i - 1] = w
#
#                if i < m - 1:
#                    y = -s * off_diag[i + 1]
#                    off_diag[i + 1] = c * off_diag[i + 1]
#
#            else:
#                if abs(d) < tol_equivalence:
#                    if off_diag[0] * d > 0:
#                        c = cp.sqrt(2) / 2
#                        s = -cp.sqrt(2) / 2
#                    else:
#                        c = s = cp.sqrt(2) / 2
#                # if off diagonal ... do nothing
#                else:
#                    b_2 = off_diag[0]
#                    if off_diag[0] * d > 0:
#                        x_0 = -cp.pi / 4
#                    else:
#                        x_0 = cp.pi / 4
#
#                    err_rel = 1
#                    iter_newton = 0
#                    while err_rel > 1e-10 and iter_newton < 1000:
#                        x_new = x_0 - cp.cos(x_0) * cp.cos(x_0) * (
#                            cp.tan(x_0) + b_2 / d
#                        )
#                        if cp.isclose(cp.linalg.norm(x_new), 0):
#                            break
#                        err_rel = cp.abs((x_new - x_0) / x_new)
#                        x_0 = x_new
#                        iter_newton += 1
#
#                    c = cp.cos(x_new / 2)
#                    s = cp.sin(x_new / 2)
#
#                    Matrix_trigonometric[i][0] = c
#                    Matrix_trigonometric[i][1] = s
#
#                    a_0 = diag[0]
#                    b_1 = off_diag[0]
#
#                    off_diag[0] = 0  # c * s * (a_0 - diag[1]) + b_1 * (c * c - s * s)
#                    diag[0] = c * c * a_0 + s * s * diag[1] - 2 * s * c * b_1
#                    diag[1] = c * c * diag[1] + s * s * a_0 + 2 * s * c * b_1
#
#        # Uncomment to compute the eigenvalue
#        # Q[:, :m] = Q[:, :m] @ Matrix_trigonometric[:m, :]
#
#        iter += 1
#        if iteration >= max_iter:
#            max_iteration_warning()
#        if abs(off_diag[m - 1]) < tol * (cp.abs(diag[m]) + cp.abs(diag[m - 1])):
#            m -= 1
#    return diag, Q
#
#
# def QR_cp(A, q0=None, tol=1e-8, max_iter=1000):
#    """
#    Compute the eigenvalues of a square matrix using the QR algorithm.
#    Done using the Lanczos algorithm to compute the tridiagonal matrix and then the QR
#    algorithm to compute the eigenvalues.
#    The implementation is done using cupy.
#
#    Args:
#        A (cp.ndarray or cpsp.spmatrix): A square matrix whose eigenvalues are to be computed.
#        q0 (cp.ndarray, optional): An initial vector for the Lanczos process. If None, a random vector is used.
#        tol (float, optional): Convergence tolerance for the QR algorithm. Default is 1e-8
#        max_iter (int, optional): Maximum number of iterations for the QR algorithm. Deault is 100.
#
#    Returns:
#        tuple: A tuple (eigenvalues, Q) where:
#            - eigenvalues (cp.ndarray): An array containing the eigenvalues of the matrix.
#            - Q (cp.ndarray): The orthogonal matrix Q from the final QR decomposition.
#
#    Raises:
#        ValueError: If the input matrix is not square.
#    """
#    _, alpha, beta = Lanczos_PRO_cp(A, q=q0, m=None, tol=1e-8)
#    return QR_method_cp(alpha, beta, tol=tol, max_iter=max_iter)
