import pandas as pd
import numpy as np
import scipy.sparse as sp
#import cupy as cp
#import cupyx.scipy.sparse as cpsp
import os
import yaml
import cProfile
from memory_profiler import memory_usage


def check_square_matrix(A):
    """
    Checks if the input matrix is a square matrix of type NumPy ndarray or SciPy sparse matrix.
    This is done to ensure that the input matrix `A` is both:
    1. Of type `np.ndarray` (NumPy array) or `scipy.sparse.spmatrix` (SciPy sparse matrix).
    2. A square matrix.

    Args:
        A (np.ndarray or sp.spmatrix or cpsp.spmatrix): The matrix to be checked.

    Raises:
        TypeError: If the input is not a NumPy array or a SciPy sparse matrix.
        ValueError: If number of rows != number of columns.
    """
    if not isinstance(A, (np.ndarray, sp.spmatrix)):
        raise TypeError(
            "Input matrix must be a NumPy array or a SciPy sparse matrix!"
        )
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square!")


def make_symmetric(A):
    """
    Ensures the input matrix is symmetric by averaging it with its transpose.

    This function first checks if the matrix is square using the `check_square_matrix` function.
    Then, it makes the matrix symmetric by averaging it with its transpose.

    Args:
        A (np.ndarray or sp.spmatrix): The input square matrix to be made symmetric.

    Returns:
        np.ndarray or sp.spmatrix: The symmetric version of the input matrix.

    Raises:
        TypeError: If the input matrix is not a NumPy array or SciPy sparse matrix.
        ValueError: If the input matrix is not square.
    """
    check_square_matrix(A)
    A_sym = (A + A.T) / 2
    return A_sym


def check_symm_square(A):
    """
    Checks if the input matrix is a square symmetric matrix of type SciPy sparse matrix.
    This is done to ensure that the input matrix `A` is all of the following:
    1. A numpy array or a scipy sparse matrix.
    2. A square matrix.
    3. Symmetric.

    Args:
        A (sp.spmatrix): The matrix to be checked.

    Raises:
        TypeError: If the input is not a NumPy array or SciPy sparse matrix.
        ValueError: If number of rows != number of columns or the matrix is not symmetric.
    """
    check_square_matrix(A)
    if isinstance(A, np.ndarray) and not np.allclose(A, A.T):
        raise ValueError("Matrix must be symmetric!")
    elif isinstance(A, sp.spmatrix) and not np.allclose(A.toarray(), A.toarray().T):
        raise ValueError("Matrix must be symmetric!")


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


#def profile_with_cupy_profiler(log_file, dim, func_name, func, *args, **kwargs):
#    """
#    Function used to profile the code using the CuPy profiler.
#    """
#    mempool = cp.get_default_memory_pool()
#
#    start_mem = mempool.used_bytes()
#
#    start = cp.cuda.Event()
#    end = cp.cuda.Event()
#
#    start.record()
#    result = func(*args, **kwargs)
#    end.record()
#    end.synchronize()
#
#    end_mem = mempool.used_bytes()
#
#    elapsed_time = cp.cuda.get_elapsed_time(start, end) / 1000
#
#    peak_mem_device = (end_mem - start_mem) / (1024**2)
#
#    print(
#        f"{func_name}: {elapsed_time:.6f} s, Peak device memory: {peak_mem_device:.6f} MB"
#    )
#
#    print(peak_mem_device)
#
#    new_entry = pd.DataFrame(
#        [[func_name, dim, elapsed_time, peak_mem_device]],
#        columns=["function", "dim", "time", "peak_device_memory"],
#    )
#
#    if os.path.exists(log_file):
#        df = pd.read_csv(log_file)
#        df = pd.concat([df, new_entry], ignore_index=True)
#    else:
#        df = new_entry
#    df.to_csv(log_file, index=False)
#
#    return result


def max_iteration_warning():
    """
    Function to warn the user that the maximum number of iteration has been reached,
    hence suggesting that the method did not converge.
    """
    print(
        "---------- Warning: the max number of iteration has been reached. ----------"
    )
    print(
        "It is likely that either the tolerance is too low or some other issue occured."
    )
