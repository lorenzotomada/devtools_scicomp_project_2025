import pandas as pd
import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cpsp
import numba
import os
import yaml
import cProfile
import cupyx.profiler as profiler
from memory_profiler import memory_usage


def check_square_matrix(A):
    """
    Checks if the input matrix is a square matrix of type NumPy ndarray or SciPy or CuPy sparse matrix.
    This is done to ensure that the input matrix `A` is both:
    1. Of type `np.ndarray` (NumPy array) or `scipy.sparse.spmatrix` (SciPy sparse matrix) or 'cupyx.scipy.sparse.spmatrix' (CuPy sparse matrix).
    2. A square matrix.

    Args:
        A (np.ndarray or sp.spmatrix or cpsp.spmatrix): The matrix to be checked.

    Raises:
        TypeError: If the input is not a NumPy array or a SciPy sparse matrix.
        ValueError: If number of rows != number of columns.
    """
    if not isinstance(A, (np.ndarray, sp.spmatrix, cpsp.spmatrix)):
        raise TypeError(
            "Input matrix must be a NumPy array or a SciPy/CuPy sparse matrix!"
        )
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square!")


def make_symmetric(A):
    """
    Ensures the input matrix is symmetric by averaging it with its transpose.

    This function first checks if the matrix is square using the `check_square_matrix` function.
    Then, it makes the matrix symmetric by averaging it with its transpose.

    Args:
        A (np.ndarray or sp.spmatrix or cpsp.spmatrix): The input square matrix to be made symmetric.

    Returns:
        np.ndarray or sp.spmatrix or cpsp.spmatrix: The symmetric version of the input matrix.

    Raises:
        TypeError: If the input matrix is not a NumPy array or SciPy or CuPy sparse matrix.
        ValueError: If the input matrix is not square.
    """
    check_square_matrix(A)
    A_sym = (A + A.T) / 2
    return A_sym


def check_symm_square(A):
    """
    Checks if the input matrix is a square symmetric matrix of type SciPy/CuPy sparse matrix.
    This is done to ensure that the input matrix `A` is all of the following:
    1. A scipy sparse matrix or CuPy sparse matrix.
    2. A square matrix.
    3. Symmetric.

    Args:
        A (sp.spmatrix or cpsp.spmatrix): The matrix to be checked.

    Raises:
        TypeError: If the input is not a SciPy or CuPy sparse matrix.
        ValueError: If number of rows != number of columns or the matrix is not symmetric.
    """
    check_square_matrix(A)
    if isinstance(A, sp.spmatrix) and not np.allclose(A.toarray(), A.toarray().T):
        raise ValueError("Matrix must be symmetric!")
    elif isinstance(A, cpsp.spmatrix) and not cp.allclose(A.toarray(), A.toarray().T):
        raise ValueError("Matrix must be symmetric!")


@numba.njit(nogil=True, parallel=True)
def power_method_numba_helper(A, max_iter=500, tol=1e-4, x=None):
    """
    Approximate the dominant eigenvalue of a square matrix using the power method.

    This helper function applies the power method to a matrix A to estimate its dominant eigenvalue.
    It returns the Rayleigh quotient, x @ A @ x, which serves as an approximation of the dominant eigenvalue.

    The function is optimized with Numba using the 'njit' decorator with nogil and parallel options.

    Args:
        A (np.ndarray): A square matrix.
        max_iter (int, optional): Maximum number of iterations to perform (default is 500).
        tol (float, optional): Tolerance for convergence based on the relative change between iterations (default is 1e-4).
        x (np.ndarray, optional): Initial guess for the eigenvector. If None, a random vector is generated.

    Returns:
        float: The approximated dominant eigenvalue of the matrix A.

    Raises:
        ValueError: If the input matrix A is not square. The check is not done using 'check_A_square_matrix' because of numba technicalities.
    """
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
    Reads a YAML configuration file and returns its contents as a dictionary.

    This function constructs the absolute path to a YAML file (by appending the '.yaml' extension
    to the provided base file name), opens the file, and parses its content using yaml.safe_load.

    Args:
        file (str): The base name of the YAML file (without the '.yaml' extension).

    Returns:
        dict: A dictionary containing the configuration parameters loaded from the YAML file.
    """
    filepath = os.path.abspath(f"{file}.yaml")
    with open(filepath, "r") as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs


def profile_with_cprofile(log_file, dim, func_name, func, *args, **kwargs):
    """
    Function used to profile the code using cProfile.
    """

    def wrapped_func(*args, **kwargs):
        mem_usage = memory_usage((func, args, kwargs))
        return func(*args, **kwargs), max(mem_usage)

    profile_output = cProfile.Profile()
    profile_output.enable()

    result, peak_mem = wrapped_func(*args, **kwargs)

    profile_output.disable()

    stats = profile_output.getstats()
    total_time = sum(stat.totaltime for stat in stats)

    print(f"{func_name}: {total_time:.6f} s, Peak memory: {peak_mem:.6f} MB")

    new_entry = pd.DataFrame(
        [[func_name, dim, total_time, peak_mem]],
        columns=["function", "dim", "time", "peak_memory"],
    )

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry
    df.to_csv(log_file, index=False)

    return result


def profile_with_cupy_profiler(log_file, dim, func_name, func, *args, **kwargs):
    """
    WORK IN PROGRESS: still not working as expected.
    Function used to profile the code using the CuPy profiler.
    """
    mempool = cp.get_default_memory_pool()

    start_mem = mempool.used_bytes()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    result = func(*args, **kwargs)
    end.record()
    end.synchronize()

    end_mem = mempool.used_bytes()

    elapsed_time = cp.cuda.get_elapsed_time(start, end) / 1000

    peak_mem_device = (end_mem - start_mem) / (1024**2)

    print(
        f"{func_name}: {elapsed_time:.6f} s, Peak device memory: {peak_mem_device:.6f} MB"
    )

    print(peak_mem_device)

    new_entry = pd.DataFrame(
        [[func_name, dim, elapsed_time, peak_mem_device]],
        columns=["function", "dim", "time", "peak_device_memory"],
    )

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry
    df.to_csv(log_file, index=False)

    return result
