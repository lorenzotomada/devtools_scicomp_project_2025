#from pyclassify.parallel_tridiag_eigen import parallel_eigen
from pyclassify import parallel_tridiag_eigen
from time import time
import numpy as np
from mpi4py import MPI
import sys


def parallel_eig(d, off_d, nprocs):

    print("inside parallel_eig")
    comm = MPI.COMM_SELF.Spawn(
    sys.executable,
    args=['parallel_tridiag_eigen.py'],
    maxprocs=nprocs
    )
    print("sending")
    comm.send(d, dest=0, tag=11)
    comm.send(off_d, dest=0, tag=12)
    print("Sent data to child, waiting for results..."); sys.stdout.flush()



    # Receive result from rank 0 of child group
    eigvals = comm.recv(source=0, tag=22)
    eigvecs= comm.recv(source=0, tag=23)
    delta_t= comm.recv(source=0, tag=24)

    comm.Disconnect()
    return eigvals, eigvecs, delta_t


 
n=1000
nprocs=4
#np.random.seed(42)
d = np.random.rand(n)*2
off_d=np.random.rand(n-1)/2
print("Starting")
file_name="Profiling_numba.txt"
t_s=time()
eigvals, eigvecs, delta_t=parallel_eig(d, off_d, nprocs)
t_e=time()

T=np.diag(d) + np.diag(off_d, 1) + np.diag(off_d, -1)
npEig_val, _ =np.linalg.eigh(T)
print(np.linalg.norm(np.abs(npEig_val-eigvals), np.inf))
with open(file_name, "a") as f:
    f.write(f"{delta_t: .4e}")

