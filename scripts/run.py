from pyclassify.utils import make_symmetric
from pyclassify.parallel_tridiag_eigen import parallel_tridiag_eigen
from time import time
from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
import scipy
import psutil
import sys
import gc


parent_comm = MPI.Comm.Get_parent()
child_comm = MPI.COMM_WORLD
rank = child_comm.Get_rank()
size = child_comm.Get_size()


if rank == 0:
    main_diag = parent_comm.recv(source=0, tag=11)
    off_diag = parent_comm.recv(source=0, tag=12)
else:
    main_diag = None
    off_diag = None


main_diag = child_comm.bcast(main_diag, root=0)
off_diag = child_comm.bcast(off_diag, root=0)


gc.collect()
proc = psutil.Process()
mem_before = proc.memory_info().rss / 1024 / 1024  # in MB


t_s = time()
eigvals, eigvecs = parallel_tridiag_eigen(
    main_diag, off_diag, comm=child_comm, min_size=1, tol_factor=1e-14
)
t_e = time()


mem_after = proc.memory_info().rss / 1024 / 1024  # in MB
delta_mem = mem_after - mem_before
total_mem = child_comm.reduce(delta_mem, op=MPI.SUM, root=0)


sys.stdout.flush()


if rank == 0:
    parent_comm.send(eigvals, dest=0, tag=22)
    parent_comm.send(eigvecs, dest=0, tag=23)
    parent_comm.send(t_e - t_s, dest=0, tag=24)
    parent_comm.send(total_mem, dest=0, tag=25)
    sys.stdout.flush()
    print("Child sent results")
parent_comm.Disconnect()
