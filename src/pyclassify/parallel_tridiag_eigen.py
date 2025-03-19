from mpi4py import MPI
import numpy as np
from time import time
from QR_cpp import QR_algorithm
from line_profiler import profile
import scipy.sparse as sp

#@profile
def deflate_eigenpairs(D, v, tol_factor=1e-12):
    norm_T = np.max(np.abs(D))  # Normalize tolerance to matrix size
    tol = tol_factor * norm_T
    keep_indices = []
    deflated_eigvals = []
    deflated_eigvecs = []
    deflated_indices = []
    reduced_dimension = len(D)

    # Zero component deflation
    e_vec = np.zeros(len(D))
    for i in range(len(D)):
        if abs(v[i]) < tol:
            deflated_eigvals.append(D[i])           
            e_vec[i] = 1.0  # Standard basis vector
            deflated_eigvecs.append(e_vec.copy())
            deflated_indices.append(i)
            e_vec[i] = 0.0
        else:
            keep_indices.append(i)

    new_order = keep_indices + deflated_indices
    reduced_dimension = len(keep_indices)

    # Create permutation matrix P (use sparse)
    n = len(D)
    P = sp.lil_array((n, n))  # Use sparse format
    P_2 = sp.eye(n, format='csr')  # Efficient multiplication format
    P_3 = sp.lil_array((n, n))  # Permutation matrix

    for new_pos, old_pos in enumerate(new_order):
        P[new_pos, old_pos] = 1  # Assign 1s to move elements accordingly

    P = P.tocsr()  # Convert to CSR format for fast multiplication
    

    D_keep = D[keep_indices]
    v_keep = v[keep_indices]

    to_check = set(np.arange(reduced_dimension, dtype=np.int64))  # Use set for fast lookup
    rotation_matrix = []
    vec_idx_list = []  # Use list instead of np.append()

    to_check_copy = list(to_check)  # Convert to list for iteration

    for i in to_check_copy[:-1]:
        if i not in to_check:
            continue  # Skip if the index was removed

        # Find duplicates in a vectorized way
        idx_duplicate_vec = np.where(np.abs(D_keep[i + 1:] - D_keep[i]) < tol)[0]
        if len(idx_duplicate_vec):
            idx_duplicate_vec += (i + 1)  # Adjust indices

            for idx_duplicate in idx_duplicate_vec:
                to_check.discard(idx_duplicate)  # O(1) removal instead of np.delete()
                
                # Compute Givens rotation parameters
                r = np.hypot(v_keep[i], v_keep[idx_duplicate])  # More stable than sqrt(x^2 + y^2)
                c = v_keep[i] / r
                s = -v_keep[idx_duplicate] / r

                v_keep[i] = r
                v_keep[idx_duplicate] = 0

                # Store transformation
                rotation_matrix.append((i, idx_duplicate, c, s))
                deflated_eigvals.append(D_keep[i])

                # Efficient eigenvector computation
                eig_vec_local = np.zeros(n)
                eig_vec_local[idx_duplicate] = c
                eig_vec_local[i] = s
                deflated_eigvecs.append(P.T @ eig_vec_local)

                vec_idx_list.append(idx_duplicate)  # Use list instead of slow np.append()

    new_order = np.concatenate((list(to_check), vec_idx_list))  # Efficient concatenation
    new_order = new_order.astype(int)

    # Apply Givens rotations
    for i, j, c, s in rotation_matrix:
        G = sp.eye(n, n, format='csr')  # Sparse identity matrix
        G[i, i] = G[j, j] = c
        G[i, j] = -s
        G[j, i] = s
        P_2 = P_2 @ G  # Sparse multiplication

    for new_pos, old_pos in enumerate(new_order):
        P_3[new_pos, old_pos] = 1  # Assign 1s

    P_3 = P_3.tocsr()
    
    to_check=[i for i in to_check]
    reduced_dimension = len(to_check)
    D_keep=D_keep[to_check]
    v_keep=v_keep[to_check]
    
    return deflated_eigvals, deflated_eigvecs, D_keep, v_keep, P_3 @ P_2 @ P



@profile
def parallel_tridiag_eigen(diag, off, comm=None, tol_factor=1e-12, min_size=1):
    """
    Computes eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
    - Uses recursive divide-and-conquer.
    - Parallelized via MPI.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = len(diag)

    # Base Case: Solve directly for small matrices
    if n <= min_size or size == 1:
        eigvals, eigvecs = QR_algorithm(diag, off)
        eigvecs=np.array(eigvecs)
        eigvals=np.array(eigvals)

        return eigvals, eigvecs

    # Divide Step: Partition into T1 and T2
    k = n // 2
    diag1, diag2 = diag[:k], diag[k:]
    off1 = off[:k-1] if k > 1 else np.array([])
    off2 = off[k:] if k < n-1 else np.array([])
    beta = off[k-1]

    diag1[-1] -= beta
    diag2[0] -= beta

    # Parallel Recursion
    left_size = size // 2 if size > 1 else 1
    color = 0 if rank < left_size else 1
    subcomm = comm.Split(color=color, key=rank)

    if color == 0:
        eigvals_left, eigvecs_left = parallel_tridiag_eigen(diag1, off1, comm=subcomm, tol_factor=tol_factor, min_size=min_size)
    else:
        eigvals_right, eigvecs_right = parallel_tridiag_eigen(diag2, off2, comm=subcomm, tol_factor=tol_factor, min_size=min_size)

    if rank == 0:
        eigvals_right = comm.recv(source=left_size, tag=77)
        eigvecs_right = comm.recv(source=left_size, tag=78)
    elif rank == left_size:
        comm.send(eigvals_right, dest=0, tag=77)
        comm.send(eigvecs_right, dest=0, tag=78)

    if rank == 0:
        # Merge Step
        n1 = len(eigvals_left)
        D = np.concatenate((eigvals_left, eigvals_right))


        v_vec = np.concatenate((eigvecs_left[-1, :],eigvecs_right[0, :]))

        

        deflated_eigvals, deflated_eigvecs, D_keep, v_keep,P = deflate_eigenpairs(D, v_vec, tol_factor)

        reduced_dim=len(D_keep)

        if D_keep.size > 0:
            M = np.diag(D_keep) + beta * np.outer(v_keep, v_keep)
            lam , _= np.linalg.eigh(M)
        else:
            lam= np.array([])

        
        # #compute v_keep again
        #v_new=np.zeros(len(v_keep))
        for k in range(lam.size):
            numerator=lam-D_keep[k]

            denominator= np.concatenate((D_keep[:k], D_keep[k+1:]))-D_keep[k]
            numerator[:-1] =numerator[:-1] / denominator
            v_keep[k]= np.sqrt(np.abs(np.prod(numerator)/beta))* np.sign(v_keep[k])

        
        # for i, j in zip(v_new , v_keep):
        #     print(i, j)



        eigenpairs = []
        for j in range(lam.size):
            y = np.zeros(D.size)
            # y[:len(D_keep)] = Z[:, j]
            y[:reduced_dim]=v_keep/(lam[j]-D_keep)
            y /= np.linalg.norm(y)
            y=P.T@y
            vec=np.concatenate((eigvecs_left@y[:n1], eigvecs_right@y[n1:]))
            vec /= np.linalg.norm(vec)

            eigenpairs.append((lam[j], vec))

        for eigval, vec in zip(deflated_eigvals, deflated_eigvecs):
            #vec = Q_block @ vec
            vec=np.concatenate((eigvecs_left@vec[:n1], eigvecs_right@vec[n1:]))
            eigenpairs.append((eigval, vec))

        eigenpairs.sort(key=lambda x: x[0])
        final_eigvals = np.array([ev for ev, _ in eigenpairs])
        final_eigvecs = np.column_stack([vec for _, vec in eigenpairs])

    else:
        final_eigvals, final_eigvecs = None, None

    final_eigvals = comm.bcast(final_eigvals, root=0)
    final_eigvecs = comm.bcast(final_eigvecs, root=0)
    return final_eigvals, final_eigvecs

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n = 2000
    if rank == 0:
        np.random.seed(42)
        main_diag = np.ones(n, dtype=np.float64)*5
        off_diag = np.ones(n-1, dtype=np.float64)*20
        # main_diag=np.random.uniform(-100, 100, n)
        # off_diag = np.random.uniform(-100, 100, n-1)
        T=np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    else:
        main_diag, off_diag = None, None

    main_diag = comm.bcast(main_diag, root=0)
    off_diag = comm.bcast(off_diag, root=0)

    t_s=time()
    eigvals, eigvecs = parallel_tridiag_eigen(main_diag, off_diag, comm=comm, min_size=1)
    t_e=time()

    print(f"Elapsed {t_e-t_s}")
    if rank == 0:

        eig_numpy, eig_vec_numpy=np.linalg.eigh(T)
        index=np.argsort(eig_numpy)
        eig_numpy=eig_numpy[index]
        eig_vec_numpy[:, index]
        index=np.argsort(eigvals)
        eigvals=eigvals[index]
        eigvecs=eigvecs[:, index]
        # print("Eigenvalue divide and conquer", eigvals)
        # print("Eigenvalue numpy", eig_numpy)

        # print("\n\n Errror eigen",eig_numpy-eigvals )

        print("Norm difference eigenaval", np.linalg.norm(eig_numpy-eigvals, np.inf))

        for count, i in enumerate(eigvecs[0, :]):
            if i<0:
                eigvecs[:, count] = (-1)*eigvecs[:, count]

        for count, i in enumerate(eig_vec_numpy[0, :]):
            if i<0:
                eig_vec_numpy[:, count] = (-1)*eig_vec_numpy[:, count]


        # print("Eigenvector solver:\n", eigvecs)
        # print("Eigenvector numpy:\n", eig_vec_numpy)
        # print("\n\n\nDifference :\n", eig_vec_numpy-eigvecs)
        print("Norm difference eigenvec", np.linalg.norm(eig_vec_numpy-eigvecs, np.inf))

        np.savez("eigen_data.npz", eigvals=eigvals, eig_numpy=eig_numpy, eigvecs=eigvecs, eig_vec_numpy=eig_vec_numpy)

        

