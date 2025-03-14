import numpy as np
import scipy.sparse as sp
from line_profiler import profile
from numpy.linalg import eig, eigh
from pyclassify.utils import check_A_square_matrix, power_method_numba_helper
from numba import jit, prange


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


from QR_cpp import QR_algorithm, Eigen_value_calculator

class EigenSolver:
    def __init__(self, A: np.ndarray, max_iter=5000, toll=1e-8):
        if not isinstance(A, (np.ndarray, sp.spmatrix)):
            raise TypeError("Input matrix must be a NumPy array or a SciPy sparse matrix!")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square!")
        if np.any(A != A.T):
            raise ValueError("Input matrix A must be symmetric.")
        
        self.A=A
        self.toll=toll
        self.max_iter=max_iter
        self.diag=None
        self.off_diag=None
        self.Q=None

        
    @jit(nopython=True, parallel=True)
    def Lanczos_PRO(self, q, A=None, m=None, toll=np.sqrt(np.finfo(float).eps)):
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

        if A==None:
            A=self.A


        if m == None:
            m = A.shape[0]

        if A.shape[0] != A.shape[1]:
            raise ValueError("Input matrix A must be square.")

        if A.shape[0] != q.shape[0]:
            raise ValueError("Input vector q must have the same size as the matrix A.")

        if np.any(A != A.T):
            raise ValueError("Input matrix A must be symmetric.")

        q = q / np.linalg.norm(q)
        # Q=np.array([q])
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
            if np.any(np.abs(q @ Q[: j - 1].T) > toll):
                partial = np.zeros((j, len(q)), dtype=np.float64)
                for i in prange(j):
                    h = 0.0
                    # Compute the dot product: h = q dot Q[i]
                    
                    h = q @Q[i]
                    # Store the contribution: h * Q[i] into the ith row
                    
                    partial[i] = h * Q[i]
            
                # Reduce the contributions (summing the partial projections)   
                q=q-np.sum(partial, axis=0)


            q = q / np.linalg.norm(q)
            Q[j] = q
            r = A @ q - beta[j - 1] * Q[j - 1]
            alpha.append(q @ r)
            r = r - alpha[j] * q
            beta.append(np.linalg.norm(r))

            if np.abs(beta[j]) < 1e-15:
                self.diag=alpha
                self.beta=beta[:-1]
                self.Q=Q
                return Q, alpha, beta[:-1]
            
        self.diag=alpha
        self.beta=beta[:-1]
        self.Q=np.array(Q)
        return Q, alpha, beta[:-1]
    
    def compute_eigenval(self, diag=None, off_diag=None):
        if diag==None and off_diag==None:
            diag=self.diag
            off_diag=self.diag
        else:
            if len(diag) != (len(off_diag) +1):
                ValueError("Mismatch  between diagonal and off diagonal size")

        return Eigen_value_calculator(diag, off_diag, self.toll, self.max_iter)



    def eig(self, diag=None, off_diag=None):
        if diag==None and off_diag==None:
            diag=self.diag
            off_diag=self.diag
        else:
            if len(diag) != (len(off_diag) +1):
                ValueError("Mismatch  between diagonal and off diagonal size")
            
            return QR_algorithm(diag, off_diag, self.toll, self.max_iter)
        
        eig, Q_triangular=QR_algorithm(diag, off_diag, self.toll, self.max_iter)
        Q_triangular=np.array(Q_triangular)
        return eig, Q_triangular @ self.Q.T
