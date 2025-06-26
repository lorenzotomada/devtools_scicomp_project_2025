from time import time
import numpy as np
import scipy
from mpi4py import MPI
import sys
import numpy as np
import argparse
from pyclassify.utils import read_config, poisson_2d_structure
from pyclassify.eigenvalues import Lanczos_PRO


seed = 8422
np.random.seed(seed)


def parallel_eig(diag, off_diag, nprocs):
    print("Spawning a communicator")
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=["scripts/run.py"], maxprocs=nprocs)

    print("Sending data to children")
    comm.send(diag, dest=0, tag=11)
    comm.send(off_diag, dest=0, tag=12)

    print("Waiting for results...")
    sys.stdout.flush()

    eigvals = comm.recv(source=0, tag=22)
    eigvecs = comm.recv(source=0, tag=23)
    delta_t = comm.recv(source=0, tag=24)
    total_mem_children = comm.recv(source=0, tag=25)
    comm.Disconnect()

    print('Data recieved!')
    return eigvals, eigvecs, delta_t, total_mem_children


def compute_eigvals(A, n_procs):
    print('Reducing using Lanczos')
    Q, diag, off_diag = Lanczos_PRO(A_np, np.ones_like(np.diag(A_np)) * 1.0)

    print('Done. Now computing eigenvalues.')
    eigvals, eigvecs, delta_t, total_mem_children = parallel_eig(diag, off_diag, n_procs)

    print("Eigenvalues computed")
    return eigvals, eigvecs, delta_t, total_mem_children


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False, help="config file:")
args = parser.parse_args()
config_file = args.config if args.config else "experiments/config"


kwargs = read_config(config_file)
dim = kwargs["dim"]
density = kwargs["density"]
n_procs = kwargs["n_processes"]


A = poisson_2d_structure(dim)
A_np = A.toarray()

# Alternatively, consider for instance:
# A = sp.random(dim, dim, density=density, format="csr") # uncomment if you want to use a random matrix instead
# or 
# eig = np.arange(1, dim + 1)
# A = np.diag(eig)
# U = scipy.stats.ortho_group.rvs(dim)

# A = U @ A @ U.T
# A = make_symmetric(A)
# A_sp = sp.csr_matrix(A)

print('---------------\nCalling Lanczos a first time to compile it...')
Q, diag, off_diag = Lanczos_PRO(A_np, np.ones_like(np.diag(A_np)) * 1.0)
print('Done! Now we compute the eigenvalues.\n---------------')

eigvals, eigvecs, delta_t, total_mem_children = compute_eigvals(A_np, n_procs)
exact_eigvals, exact_eigvecs = np.linalg.eig(A.toarray())
print('---------------')
sorted_indices = np.argsort(exact_eigvals)
exact_eigvals = exact_eigvals[sorted_indices]
exact_eigvecs = exact_eigvecs[:, sorted_indices]

max_error = np.max(np.abs(exact_eigvals-eigvals))
print(f'The maximum error between real and computed eigenvalues is {max_error}')

if max_error < 1e-8:
    print('Pretty small, huh?')