import numpy as np
import scipy.sparse as sp
import scipy
import os
import psutil
import gc
import yaml

# import cProfile
# from memory_profiler import memory_usage
# import cupy as cp
# import cupyx.scipy.sparse as cpsp


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
        raise TypeError("Input matrix must be a NumPy array or a SciPy sparse matrix!")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square!")


def make_symmetric(A):
    """
    Ensures that the input matrix is symmetric by averaging it with its transpose.

    This function performs the following steps:
    1. Checks if the matrix is a square matrix using `check_square_matrix`.
    2. Computes the symmetric version of the matrix using the formula: (A + A.T) / 2.

    Args:
        A (np.ndarray or sp.spmatrix): The input matrix to be symmetrized.

    Returns:
        np.ndarray or sp.spmatrix: The symmetric version of the input matrix.

    Raises:
        TypeError: If the input is not a NumPy array or a SciPy sparse matrix.
        ValueError: If the matrix is not square.
    """

    check_square_matrix(A)
    A_sym = (A + A.T) / 2
    return A_sym


def check_symm_square(A):
    """
    Checks if the input matrix is a square symmetric matrix of type NumPy ndarray or SciPy sparse matrix. It uses check_square_matrix.

    This function performs the following steps:
    1. Verifies the input type is either `np.ndarray` or `sp.spmatrix`.
    2. Checks that the matrix is square.
    3. Validates that the matrix is symmetric.

    Args:
        A (np.ndarray or sp.spmatrix): The matrix to be validated.

    Raises:
        TypeError: If the input is not a NumPy array or a SciPy sparse matrix.
        ValueError: If the matrix is not square or not symmetric.
    """
    check_square_matrix(A)
    if isinstance(A, np.ndarray) and not np.allclose(A, A.T):
        raise ValueError("Matrix must be symmetric!")
    elif isinstance(A, sp.spmatrix) and not np.allclose(A.toarray(), A.toarray().T):
        raise ValueError("Matrix must be symmetric!")


def read_config(file: str) -> dict:
    """
    Reads a YAML configuration file and loads its contents into a dictionary.

    This function performs the following steps:
    1. Constructs the absolute path to the YAML file by appending '.yaml' to the given base name.
    2. Opens and parses the file using `yaml.safe_load`.

    Args:
        file (str): The base name of the YAML file (without extension).

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    filepath = os.path.abspath(f"{file}.yaml")
    with open(filepath, "r") as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs


def max_iteration_warning():
    """
    Prints a warning message indicating that the maximum number of iterations has been reached.

    This function is used to alert the user that the iterative method likely did not converge,
    and suggests that a lower tolerance or alternative approach may be needed.
    """
    print(
        "---------- Warning: the max number of iteration has been reached. ----------"
    )
    print(
        "It is likely that either the tolerance is too low or some other issue occured."
    )


def poisson_2d_structure(n, k=None):
    """
    Constructs a sparse matrix of size n x n with the same 5-diagonal structure
    as the 2D Poisson matrix (main, ±1, ±k). Used for testing structure, not actual PDEs: in real
    life, this function would return a matrix of size $n^2$.

    Args:
        n (int): Size of the square matrix.
        k (int or None): The offset for the "long-range" diagonals. If None, uses k = int(sqrt(n)).

    Returns:
        scipy.sparse.spmatrix: A sparse matrix with 5 diagonals for testing.
    """

    if k is None:
        k = max(1, int(n**0.5))  # simulate ±sqrt(n) diagonal positions

    diagonals = [
        -1 * np.ones(n - 1),
        -1 * np.ones(n - k),
        4 * np.ones(n),
        -1 * np.ones(n - k),
        -1 * np.ones(n - 1),
    ]
    offsets = [-1, -k, 0, k, 1]

    return sp.diags(diagonals, offsets, shape=(n, n), format="csr")


def profile_numpy_eigvals(A):
    """
    Profiles the memory usage of computing eigenvalues and eigenvectors using NumPy.

    This function performs the following steps:
    1. Measures memory before the eigendecomposition.
    2. Computes eigenvalues and eigenvectors using `np.linalg.eigh`.
    3. Measures memory after the computation.
    4. Returns the difference in memory usage.

    Args:
        A (np.ndarray): A symmetric NumPy array.

    Returns:
        float: Memory used in MB during the computation.
    """
    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    # NumPy symmetric eig solver
    eigvals, eigvecs = np.linalg.eigh(A)

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    delta_mem = mem_after - mem_before
    return delta_mem


def profile_scipy_eigvals(A):
    """
    Profiles the memory usage of computing eigenvalues and eigenvectors using SciPy.

    This function performs the following steps:
    1. Measures memory before the eigendecomposition.
    2. Computes eigenvalues and eigenvectors using `scipy.linalg.eigh`.
    3. Measures memory after the computation.
    4. Returns the difference in memory usage.

    Args:
        A (np.ndarray): A symmetric NumPy array.

    Returns:
        float: Memory used in MB during the computation.
    """
    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    # SciPy symmetric eig solver
    eigvals, eigvecs = scipy.linalg.eigh(A)

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    delta_mem = mem_after - mem_before
    return delta_mem


######### OUTDATED FUNCTIONS: working but no longer used #########
#
# def profile_with_cprofile(log_file, dim, func_name, func, *args, **kwargs):
#    """
#    Function used to profile the code using cProfile.
#    """
#
#    def wrapped_func(*args, **kwargs):
#        mem_usage = memory_usage((func, args, kwargs))
#        return func(*args, **kwargs), max(mem_usage)
#
#    profile_output = cProfile.Profile()
#    profile_output.enable()
#
#    result, peak_mem = wrapped_func(*args, **kwargs)
#
#    profile_output.disable()
#
#    stats = profile_output.getstats()
#    total_time = sum(stat.totaltime for stat in stats)
#
#    print(f"{func_name}: {total_time:.6f} s, Peak memory: {peak_mem:.6f} MB")
#
#    new_entry = pd.DataFrame(
#        [[func_name, dim, total_time, peak_mem]],
#        columns=["function", "dim", "time", "peak_memory"],
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
#
#
# def profile_with_cupy_profiler(log_file, dim, func_name, func, *args, **kwargs):
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
#
#
#
# def get_memory_usage_mb():
#     """
#     Function to get the current process memory usage in MB.
#     """
#     return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
#
#
# def mpi_profiled(func):
#     """
#     Decorator to profile time and memory across MPI ranks.
#     """
#
#     def wrapper(*args, **kwargs):
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#
#         mem_before = get_memory_usage_mb()
#         start = time.time()
#         result = func(*args, **kwargs)
#         end = time.time()
#         mem_after = get_memory_usage_mb()
#
#         delta_mem = mem_after - mem_before
#         duration = end - start
#
#         # Now we gather all the info
#         # We will be measuring the total memory used by all processes and the time spent by the process that took the longest to complete
#         all_mem = comm.gather(delta_mem, root=0)
#         all_time = comm.gather(duration, root=0)
#
#         if (
#             rank == 0
#         ):  # we only return if the rank is 0 (no need to return the same information multiple times)
#             return {
#                 "result": result,
#                 "memory_total": sum(all_mem),
#                 "memory_max": max(all_mem),
#                 "time": max(all_time),
#             }
#         else:
#             return None
#
#     return wrapper
#
#
# def profile_serial(func, *args, **kwargs):
#     """
#     Instead, this function is used to profile regular functions that do not use MPI internally.
#     """
#     mem_before = get_memory_usage_mb()
#     start = time.time()
#     result = func(*args, **kwargs)
#     end = time.time()
#     mem_after = get_memory_usage_mb()
#     return {
#         "result": result,
#         "memory_total": mem_after - mem_before,
#         "memory_max": mem_after - mem_before,
#         "time": end - start,
#     }
