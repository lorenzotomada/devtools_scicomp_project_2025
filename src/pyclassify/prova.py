import numpy as np
from time import time
from numba import jit, prange


@jit(nopython=True, parallel=True)
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

            return Q, alpha, beta[:-1]
    return Q, alpha, beta[:-1]


matrix = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [2, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [3, 12, 20, 21, 22, 23, 24, 25, 26, 27],
        [4, 13, 21, 28, 29, 30, 31, 32, 33, 34],
        [5, 14, 22, 29, 35, 36, 37, 38, 39, 40],
        [6, 15, 23, 30, 36, 41, 42, 43, 44, 45],
        [7, 16, 24, 31, 37, 42, 46, 47, 48, 49],
        [8, 17, 25, 32, 38, 43, 47, 50, 51, 52],
        [9, 18, 26, 33, 39, 44, 48, 51, 53, 54],
        [10, 19, 27, 34, 40, 45, 49, 52, 54, 55],
    ],
    dtype=np.float64,
)

# Define the initial guess vector (all ones, double precision)
initial_guess = np.ones(10, dtype=np.float64)
print(Lanczos_PRO(matrix, initial_guess, 10))
matrix = np.random.rand(3000, 3000)
matrix = 0.5 * (matrix + matrix.T)

t_s = time()
Lanczos_PRO(matrix, np.ones(3000), 3000)
t_e = time()
print(f"Elapsed {t_s-t_e: .4e}")
