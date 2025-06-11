from mpi4py import MPI
import psutil
import time
import os


def get_memory_usage_mb():
    """
    Function to get the current process memory usage in MB.
    """
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def mpi_profiled(func):
    """
    Decorator to profile time and memory across MPI ranks.
    """

    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        mem_before = get_memory_usage_mb()
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        mem_after = get_memory_usage_mb()

        delta_mem = mem_after - mem_before
        duration = end - start

        # Now we gather all the info
        # We will be measuring the total memory used by all processes and the time spent by the process that took the longest to complete
        all_mem = comm.gather(delta_mem, root=0)
        all_time = comm.gather(duration, root=0)

        if (
            rank == 0
        ):  # we only return if the rank is 0 (no need to return the same information multiple times)
            return {
                "result": result,
                "memory_total": sum(all_mem),
                "memory_max": max(all_mem),
                "time": max(all_time),
            }
        else:
            return None

    return wrapper


def profile_serial(func, *args, **kwargs):
    """
    Instead, this function is used to profile regular functions that do not use MPI internally.
    """
    mem_before = get_memory_usage_mb()
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    mem_after = get_memory_usage_mb()
    return {
        "result": result,
        "memory_total": mem_after - mem_before,
        "memory_max": mem_after - mem_before,
        "time": end - start,
    }
