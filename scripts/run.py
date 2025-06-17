from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
import scipy
from pyclassify.utils import make_symmetric
from pyclassify.parallel_tridiag_eigen import parallel_tridiag_eigen
import psutil
import time
import sys
import gc

comm = MPI.Comm.Get_parent()
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

if rank == 0:
    main_diag = comm.recv(source=0, tag=11)
    off_diag = comm.recv(source=0, tag=12)
else:
    main_diag = None
    off_diag = None

main_diag = MPI.COMM_WORLD.bcast(main_diag, root=0)
off_diag = MPI.COMM_WORLD.bcast(off_diag, root=0)

gc.collect()
proc = psutil.Process()
mem_before = proc.memory_info().rss / 1024 / 1024  # in MB

t_s = time.time()
eigvals, eigvecs = parallel_tridiag_eigen(
   main_diag, off_diag, comm=MPI.COMM_WORLD, min_size=1, tol_factor=1e-10
)
t_e = time.time()

mem_after = proc.memory_info().rss / 1024 / 1024  # in MB
delta_mem = mem_after - mem_before

total_mem = MPI.COMM_WORLD.reduce(delta_mem, op=MPI.SUM, root=0)

print("Function run")
sys.stdout.flush()

if rank == 0:
    comm.send(eigvals, dest=0, tag=22)
    comm.send(eigvecs, dest=0, tag=23)
    comm.send(t_e - t_s, dest=0, tag=24)
    comm.send(total_mem, dest=0, tag=25)
    print("Child sent results")
    sys.stdout.flush()
    comm.Disconnect()
