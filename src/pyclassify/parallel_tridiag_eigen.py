from mpi4py import MPI
import numpy as np
from time import time
from pyclassify.cxx_utils import (
    QR_algorithm,
    secular_solver_cxx,
    deflate_eigenpairs_cxx,
)
from pyclassify.zero_finder import secular_solver_python as secular_solver
from line_profiler import profile, LineProfiler
import scipy.sparse as sp
from line_profiler import LineProfiler
import scipy


profile = LineProfiler()


def check_column_directions(A, B):
    n = A.shape[1]
    for i in range(n):
        # Normalize the vectors
        a = A[:, i]
        b = B[:, i]
        dot = np.dot(a, b)
        # Should be close to 1 or -1
        if dot < 0:
            B[:, i] = -B[:, i]


@profile
def deflate_eigenpairs(D, v, beta, tol_factor=1e-12):
    """
    Applying the deflation step to the divide and conquer algorithm to reduce the size of
    the system to be solved and find the trivial eigenvalues and eigenvectors.
    Notice that we cannot use jit since we work with scipy sparse matrices.
    Input:
        -D: element on the diagonal.
        -v: rank one update vector
        -beta: scalar value that multiplies the rank one update

    Output:
        -deflated_eigvals: Vector containing the trivial eigenvalue
        -deflated_eigvecs: Matrix containing the trivial eigenvectors
        -D_keep: element on the diagonal after the deflation.
        -v_keep: rank one update vector after the deflation
        -P_3 @ P_2 @ P: Permutation matrix
    """

    norm_T = np.linalg.norm(
        np.diag(D) + beta * np.outer(v, v)
    )  # Normalize tolerance to matrix size
    # abs_v=np.abs(v)
    # norm_T=0.5*(np.max(np.abs(D)) + beta * np.max(abs_v)*np.sum(abs_v))
    tol = tol_factor * norm_T
    keep_indices = []
    deflated_eigvals = np.zeros_like(D)
    deflated_eigvecs = []
    deflated_indices = []
    reduced_dimension = len(D)

    # Zero component deflation
    e_vec = np.zeros(len(D))
    j = 0
    for i in range(len(D)):
        if abs(v[i]) < tol:
            deflated_eigvals[j] = D[i]
            e_vec[i] = 1.0  # Standard basis vector
            deflated_eigvecs.append(e_vec.copy())
            deflated_indices.append(i)
            e_vec[i] = 0.0
            j += 1
        else:
            keep_indices.append(i)

    new_order = keep_indices + deflated_indices
    reduced_dimension = len(keep_indices)

    # Create permutation matrix P (use sparse)
    n = len(D)
    P = sp.lil_array((n, n))  # Use sparse format
    P_2 = sp.eye(n, format="csr")  # Efficient multiplication format
    P_3 = sp.lil_array((n, n))  # Permutation matrix

    for new_pos, old_pos in enumerate(new_order):
        P[new_pos, old_pos] = 1  # Assign 1s to move elements accordingly

    P = P.tocsr()  # Convert to CSR format for fast multiplication

    D_keep = D[keep_indices]
    v_keep = v[keep_indices]

    to_check = set(
        np.arange(reduced_dimension, dtype=np.int64)
    )  # Use set for fast lookup
    rotation_matrix = []
    vec_idx_list = []  # Use list instead of np.append()

    to_check_copy = list(to_check)  # Convert to list for iteration

    for i in to_check_copy[:-1]:
        if i not in to_check:
            continue  # Skip if the index was removed

        # Find duplicates in a vectorized way
        idx_duplicate_vec = np.where(np.abs(D_keep[i + 1 :] - D_keep[i]) < tol)[0]
        if len(idx_duplicate_vec):
            idx_duplicate_vec += i + 1  # Adjust indices

            for idx_duplicate in idx_duplicate_vec:
                to_check.discard(idx_duplicate)  # O(1) removal instead of np.delete()

                # Compute Givens rotation parameters
                r = np.hypot(
                    v_keep[i], v_keep[idx_duplicate]
                )  # More stable than sqrt(x^2 + y^2)
                c = v_keep[i] / r
                s = -v_keep[idx_duplicate] / r

                v_keep[i] = r
                v_keep[idx_duplicate] = 0

                # Store transformation
                rotation_matrix.append((i, idx_duplicate, c, s))
                deflated_eigvals[j] = D_keep[i]
                j += 1

                # Efficient eigenvector computation
                eig_vec_local = np.zeros(n)
                eig_vec_local[idx_duplicate] = c
                eig_vec_local[i] = s
                deflated_eigvecs.append(P.T @ eig_vec_local)

                vec_idx_list.append(
                    idx_duplicate
                )  # Use list instead of slow np.append()

    new_order = np.concatenate(
        (list(to_check), vec_idx_list)
    )  # Efficient concatenation
    new_order = new_order.astype(int)
    deflated_eigvals = deflated_eigvals[:j]

    # Apply Givens rotations
    for i, j, c, s in rotation_matrix:
        G = sp.eye(n, n, format="csr")  # Sparse identity matrix
        G[i, i] = G[j, j] = c
        G[i, j] = -s
        G[j, i] = s
        P_2 = P_2 @ G  # Sparse multiplication

    for new_pos, old_pos in enumerate(new_order):
        P_3[new_pos, old_pos] = 1  # Assign 1s

    P_3 = P_3.tocsr()

    to_check = [i for i in to_check]
    reduced_dimension = len(to_check)
    D_keep = D_keep[to_check]
    v_keep = v_keep[to_check]

    return deflated_eigvals, np.array(deflated_eigvecs), D_keep, v_keep, P_3 @ P_2 @ P


def find_interval_extreme(total_dimension, n_processor):
    """
    Computes the intervals for vector for being scattered.
        Input:
            -total_dimension: the dimension of the vector that has to be splitted
            -n_processor:     the number of processor to which the scatter vector has to be sent

    """

    base = total_dimension // n_processor
    rest = total_dimension % n_processor

    counts = np.array(
        [base + 1 if i < rest else base for i in range(n_processor)], dtype=int
    )
    displs = np.insert(np.cumsum(counts), 0, 0)[:-1]

    return counts, displs


@profile
def parallel_tridiag_eigen(
    diag,
    off,
    comm=None,
    tol_factor=1e-16,
    min_size=1,
    depth=0,
    profiler=None,
    tol_QR=1e-8,
    max_iterQR=5000,
):
    """
    Computes eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
        Input:
        -diag: diagonal part of the tridiagonal matrix
        -off: off diagonal part of the tridiagonal matrix
        -comm: MPI communicator
        -tol_factor: tollerance for the deflating step

        Output:
        -final_eigvals: return the eigenvalues of the tridiagonal matrix
        -final_eigvecs: return the eigenvectors of the tridiagonal matrix

    """

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    current_rank = MPI.COMM_WORLD.Get_rank()
    n = len(diag)
    prof_filename = f"Profile_folder/profile.rank{current_rank}.depth{depth}.lprof"

    if n <= min_size or size == 1:
        eigvals, eigvecs = QR_algorithm(diag, off, 1e-16, max_iterQR)
        eigvecs = np.array(eigvecs)
        eigvals = np.array(eigvals)

        index_sort = np.argsort(eigvals)
        eigvecs = eigvecs[:, index_sort]
        eigvals = eigvals[index_sort]
        return eigvals, eigvecs

    k = n // 2
    diag1, diag2 = diag[:k].copy(), diag[k:].copy()
    off1 = off[: k - 1].copy() if k > 1 else np.array([])
    off2 = off[k:] if k < n - 1 else np.array([])
    beta = off[k - 1].copy()

    diag1[-1] -= beta
    diag2[0] -= beta

    # Parallel Recursion
    left_size = size // 2 if size > 1 else 1
    color = 0 if rank < left_size else 1
    subcomm = comm.Split(color=color, key=rank)

    if color == 0:
        eigvals_left, eigvecs_left = parallel_tridiag_eigen(
            diag1,
            off1,
            comm=subcomm,
            tol_factor=tol_factor,
            min_size=min_size,
            depth=depth + 1,
            profiler=profiler,
        )
        eigvals_right = None
        eigvecs_right = None
    else:
        eigvals_right, eigvecs_right = parallel_tridiag_eigen(
            diag2,
            off2,
            comm=subcomm,
            tol_factor=tol_factor,
            min_size=min_size,
            depth=depth + 1,
            profiler=profiler,
        )
        eigvals_left = None
        eigvecs_left = None

    # 1) Identify the two “root” ranks in MPI.COMM_WORLD
    left_size = size // 2 if size > 1 else 1
    root_left = 0
    root_right = left_size
    other_root = root_right if color == 0 else root_left

    # now exchange between the two roots
    if subcomm.Get_rank() == 0:
        send_data = (
            (eigvals_left, eigvecs_left)
            if color == 0
            else (eigvals_right, eigvecs_right)
        )
        recv_data = comm.sendrecv(
            send_data, dest=other_root, sendtag=depth, source=other_root, recvtag=depth
        )
        # unpack
        if color == 0:
            eigvals_right, eigvecs_right = recv_data
        else:
            eigvals_left, eigvecs_left = recv_data

    eigvals_left = subcomm.bcast(eigvals_left, root=0)
    eigvecs_left = subcomm.bcast(eigvecs_left, root=0)
    eigvals_right = subcomm.bcast(eigvals_right, root=0)
    eigvecs_right = subcomm.bcast(eigvecs_right, root=0)

    # if rank == 0:
    #     eigvals_right = comm.recv(source=left_size, tag=77)
    #     eigvecs_right = comm.recv(source=left_size, tag=78)
    # elif rank == left_size:
    #     comm.send(eigvals_right, dest=0, tag=77)
    #     comm.send(eigvecs_right, dest=0, tag=78)

    if rank == 0:

        # Merge Step
        n1 = len(eigvals_left)
        D = np.concatenate((eigvals_left, eigvals_right))
        D_size = D.size
        v_vec = np.concatenate((eigvecs_left[-1, :], eigvecs_right[0, :]))
        deflated_eigvals, deflated_eigvecs, D_keep, v_keep, P = deflate_eigenpairs_cxx(
            D, v_vec, beta, tol_factor
        )
        D_keep = np.array(D_keep)

        reduced_dim = len(D_keep)

        if D_keep.size > 0:
            idx = np.argsort(D_keep)
            idx_inv = np.arange(0, reduced_dim)
            idx_inv = idx_inv[idx]

            lam, changing_position, delta = secular_solver_cxx(
                beta, D_keep[idx], v_keep[idx], np.arange(reduced_dim)
            )
            lam = np.array(lam)
            delta = np.array(delta)
            changing_position = np.array(changing_position)
            # #diff=lam_s-lam
        else:
            lam = np.array([])

        counts, displs = find_interval_extreme(reduced_dim, size)

    else:
        counts = None
        displs = None
        lam = None
        D_keep = None
        v_keep = None
        delta = None
        reduced_dim = None
        D_size = None
        changing_position = None
        type_lam = None
        type_D = None
        P = None
        idx_inv = None
        n1 = None
        deflated_eigvals = None
        deflated_eigvecs = None

    counts = comm.bcast(counts, root=0)
    displs = comm.bcast(displs, root=0)
    lam = comm.bcast(lam, root=0)
    D_keep = comm.bcast(D_keep, root=0)
    v_keep = comm.bcast(v_keep, root=0)
    my_count = counts[rank]
    type_lam = comm.bcast(lam.dtype, root=0)
    # type_D=comm.bcast(D_keep.dtype, root=0)
    lam_buffer = np.empty(my_count, dtype=type_lam)
    # D_buffer=np.empty(my_count, dtype=type_D)
    P = comm.bcast(P, root=0)
    D_size = comm.bcast(D_size)
    changing_position = comm.bcast(changing_position, root=0)
    delta = comm.bcast(delta, root=0)
    idx_inv = comm.bcast(idx_inv, root=0)
    n1 = comm.bcast(n1, root=0)
    reduced_dim = comm.bcast(reduced_dim, root=0)

    # map numpy dtype → MPI datatype
    if lam.dtype == np.float64:
        mpi_type = MPI.DOUBLE
    elif lam.dtype == np.float32:
        mpi_type = MPI.FLOAT
    elif lam.dtype == np.int32:
        mpi_type = MPI.INT
    elif lam.dtype == np.int64:
        mpi_type = MPI.LONG
    else:
        raise ValueError(f"Unsupported dtype for Scatterv: {lam.dtype}")

    # now do the scatterv
    comm.Scatterv(
        [lam, counts, displs, mpi_type],  # send tuple, only root’s lam is used here
        lam_buffer,  # recvbuf on every rank
        root=0,
    )

    # # Scatterv with matching MPI datatype
    # comm.Scatterv([lam, counts, displs, MPI._typedict[type_lam] ],
    #           lam_buffer, root=0)
    # comm.Scatterv([D_keep, counts, displs, MPI._typedict[type_lam] ],
    #           D_buffer, root=0)

    initial_point = displs[rank]

    for k_rel in range(lam_buffer.size):
        k = k_rel + initial_point
        numerator = lam - D_keep[k]
        denominator = np.concatenate((D_keep[:k], D_keep[k + 1 :])) - D_keep[k]
        numerator[:-1] = numerator[:-1] / denominator
        v_keep[k] = np.sqrt(np.abs(np.prod(numerator) / beta)) * np.sign(v_keep[k])

    # eigenpairs = []

    eig_vecs = np.empty((D_size, my_count), dtype=type_lam)
    eig_val = np.empty(my_count, dtype=type_lam)

    for j_rel in range(lam_buffer.size):
        y = np.zeros(D_size)
        # y[:reduced_dim]=v_keep/(lam[j]-D_keep)
        # y /= np.linalg.norm(y)
        j = j_rel + initial_point
        diff = lam[j] - D_keep
        diff[idx_inv[changing_position[j]]] = delta[j]
        y[:reduced_dim] = v_keep / (diff)
        y_norm = np.linalg.norm(y)
        if y_norm > 1e-15:
            y /= y_norm

        y = P.T @ y
        vec = np.concatenate((eigvecs_left @ y[:n1], eigvecs_right @ y[n1:]))
        vec /= np.linalg.norm(vec)
        eig_vecs[:, j_rel] = vec
        eig_val[j_rel] = lam[j]
        # eigenpairs.append((lam[j], vec))

    if reduced_dim < D_size:

        if rank == 0:
            le_deflation = len(deflated_eigvals)
            counts, displs = find_interval_extreme(le_deflation, size)

        counts = comm.bcast(counts, root=0)
        displs = comm.bcast(displs, root=0)
        my_count = counts[rank]

        deflated_eigvals_buffer = np.empty(my_count, dtype=type_lam)
        if rank == 0:
            char = deflated_eigvals.dtype.char
            type_eig = deflated_eigvals.dtype
        else:
            char = None
            type_eig = None

        # now everyone learns the character code:
        char = comm.bcast(char, root=0)
        type_eig = comm.bcast(type_eig, root=0)
        comm.Scatterv(
            [deflated_eigvals, counts, displs, MPI._typedict[char]],
            deflated_eigvals_buffer,
            root=0,
        )
        if rank == 0:
            _, k = deflated_eigvecs.shape
        else:
            mat = None
            k = None
        k = comm.bcast(k, root=0)
        # each row of `mat` is one deflated vec
        sendcounts = [c * k for c in counts]
        senddispls = [d * k for d in displs]
        deflated_eigvecs_buffer = np.empty((my_count, k), dtype=type_eig)
        if rank == 0:
            flat_send = deflated_eigvecs.copy().flatten()  # shape (M*k,)
            sendbuf = [flat_send, sendcounts, senddispls, MPI._typedict[char]]
        else:
            sendbuf = None

        # now scatter to everyone
        comm.Scatterv(
            sendbuf,  # only meaningful on rank 0
            deflated_eigvecs_buffer,  # each rank’s recv‐buffer of length k × my_count
            root=0,
        )

        # local_final_vecs = np.empty((k, my_count), dtype=deflated_eigvecs.dtype)
        for i in range(my_count):
            small_vec = deflated_eigvecs_buffer[i]
            # apply the two block Q’s:
            left_part = eigvecs_left @ small_vec[:n1]
            right_part = eigvecs_right @ small_vec[n1:]
            local_final_vecs = np.concatenate((left_part, right_part))
            local_final_vecs = local_final_vecs.reshape(k, 1)
            eig_val = np.append(eig_val, deflated_eigvals_buffer[i])
            eig_vecs = np.concatenate([eig_vecs, local_final_vecs], axis=1)

    # 1) Each rank computes its local length:
    local_count = eig_val.size  # or however many elements you’ll send

    # 2) Everyone exchanges counts via allgather:
    #    this returns a Python list of length `size` on every rank
    recvcounts = comm.allgather(local_count)

    # # 1) Gather all the counts to rank 0
    # counts = comm.gather(local_count, root=0)

    # # 2) Broadcast that list from rank 0 back to everyone
    # recvcounts = comm.bcast(counts, root=0)

    final_eig_val = np.empty(D_size, dtype=eig_val.dtype)

    displs = np.append([0], np.cumulative_sum(recvcounts[:-1]).astype(int))

    mpi_t = MPI._typedict[eig_val.dtype.char]
    comm.Allgatherv([eig_val, mpi_t], [final_eig_val, recvcounts, displs, mpi_t])

    # 1) Flatten local eigenvector block
    #    eig_vecs has shape (D_size, local_count)
    local_flat = eig_vecs.T.flatten()

    # 2) Build sendcounts & displacements for the flattened arrays
    sendcounts_vecs = [c * D_size for c in recvcounts]
    senddispls_vecs = [d * D_size for d in displs]

    # 3) Allocate full receive buffer on every rank
    flat_all = np.empty(sum(sendcounts_vecs), dtype=eig_vecs.dtype)

    # 4) Perform the all-gather-variable-counts
    mpi_tvec = MPI._typedict[eig_vecs.dtype.char]
    comm.Allgatherv(
        [local_flat, mpi_tvec],  # sendbuf
        [flat_all, sendcounts_vecs, senddispls_vecs, mpi_tvec],  # recvbuf spec
    )

    # 5) Reshape on every rank (or just on rank 0 if you prefer)
    #    total_pairs == sum(recvcounts)
    final_eig_vecs = flat_all.reshape(D_size, D_size)
    final_eig_vecs = final_eig_vecs.T
    index_sort = np.argsort(final_eig_val)
    final_eig_vecs = final_eig_vecs[:, index_sort]
    final_eig_val = final_eig_val[index_sort]
    # if rank==0:
    #     print(final_eig_val)
    #     print(final_eig_vecs)
    return final_eig_val, final_eig_vecs

    # for eigval, vec in zip(deflated_eigvals, deflated_eigvecs):
    #     # vec = Q_block @ vec
    #     vec = np.concatenate((eigvecs_left @ vec[:n1], eigvecs_right @ vec[n1:]))
    #     eigenpairs.append((eigval, vec))
    # eigenpairs.sort(key=lambda x: x[0])
    # final_eigvals = np.array([ev for ev, _ in eigenpairs])
    # final_eigvecs = np.column_stack([vec for _, vec in eigenpairs])
    # else:
    #     final_eigvals, final_eigvecs = None, None

    # final_eigvals = comm.bcast(final_eigvals, root=0)
    # final_eigvecs = comm.bcast(final_eigvecs, root=0)

    # return final_eigvals, final_eigvecs


# @profile
# def parallel_tridiag_eigen(
#     diag,
#     off,
#     comm=None,
#     tol_factor=1e-16,
#     min_size=1,
#     depth=0,
#     profiler=None,
#     tol_QR=1e-8,
#     max_iterQR=5000,
# ):
#     """
#     Computes eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
#         Input:
#         -diag: diagonal part of the tridiagonal matrix
#         -off: off diagonal part of the tridiagonal matrix
#         -comm: MPI communicator
#         -tol_factor: tollerance for the deflating step

#         Output:
#         -final_eigvals: return the eigenvalues of the tridiagonal matrix
#         -final_eigvecs: return the eigenvectors of the tridiagonal matrix

#     """

#     if comm is None:
#         comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     current_rank = MPI.COMM_WORLD.Get_rank()
#     n = len(diag)
#     prof_filename = f"Profile_folder/profile.rank{current_rank}.depth{depth}.lprof"


#     if n <= min_size or size == 1:
#         eigvals, eigvecs = QR_algorithm(diag, off, tol_QR, max_iterQR)
#         eigvecs = np.array(eigvecs)
#         eigvals = np.array(eigvals)

#         index_sort = np.argsort(eigvals)
#         eigvecs = eigvecs[:, index_sort]
#         eigvals = eigvals[index_sort]
#         return eigvals, eigvecs

#     k = n // 2
#     diag1, diag2 = diag[:k].copy(), diag[k:].copy()
#     off1 = off[: k - 1].copy() if k > 1 else np.array([])
#     off2 = off[k:] if k < n - 1 else np.array([])
#     beta = off[k - 1].copy()

#     diag1[-1] -= beta
#     diag2[0] -= beta

#     # Parallel Recursion
#     left_size = size // 2 if size > 1 else 1
#     color = 0 if rank < left_size else 1
#     subcomm = comm.Split(color=color, key=rank)

#     if color == 0:
#         eigvals_left, eigvecs_left = parallel_tridiag_eigen(
#             diag1,
#             off1,
#             comm=subcomm,
#             tol_factor=tol_factor,
#             min_size=min_size,
#             depth=depth + 1,
#             profiler=profiler,
#         )
#     else:
#         eigvals_right, eigvecs_right = parallel_tridiag_eigen(
#             diag2,
#             off2,
#             comm=subcomm,
#             tol_factor=tol_factor,
#             min_size=min_size,
#             depth=depth + 1,
#             profiler=profiler,
#         )


#     if rank == 0:
#         eigvals_right = comm.recv(source=left_size, tag=77)
#         eigvecs_right = comm.recv(source=left_size, tag=78)
#     elif rank == left_size:
#         comm.send(eigvals_right, dest=0, tag=77)
#         comm.send(eigvecs_right, dest=0, tag=78)


#     if rank == 0:

#         # Merge Step
#         n1 = len(eigvals_left)
#         D = np.concatenate((eigvals_left, eigvals_right))
#         v_vec = np.concatenate((eigvecs_left[-1, :], eigvecs_right[0, :]))
#         deflated_eigvals, deflated_eigvecs, D_keep, v_keep, P = deflate_eigenpairs_cxx(
#             D, v_vec, beta, tol_factor
#         )
#         D_keep=np.array(D_keep)

#         reduced_dim = len(D_keep)

#         if D_keep.size > 0:
#             idx = np.argsort(D_keep)
#             idx_inv = np.arange(0, reduced_dim)
#             idx_inv = idx_inv[idx]

#             lam, changing_position, delta = secular_solver_cxx(
#                 beta, D_keep[idx], v_keep[idx], np.arange(reduced_dim)
#             )
#             lam = np.array(lam)
#             delta=np.array(delta)
#             changing_position=np.array(changing_position)
#             # #diff=lam_s-lam
#         else:
#             lam = np.array([])


#         for k in range(lam.size):
#             numerator = lam - D_keep[k]
#             denominator = np.concatenate((D_keep[:k], D_keep[k + 1 :])) - D_keep[k]
#             numerator[:-1] = numerator[:-1] / denominator
#             v_keep[k] = np.sqrt(np.abs(np.prod(numerator) / beta)) * np.sign(v_keep[k])

#         eigenpairs = []

#         for j in range(lam.size):
#             y = np.zeros(D.size)
#             # y[:reduced_dim]=v_keep/(lam[j]-D_keep)
#             # y /= np.linalg.norm(y)

#             diff = lam[j] - D_keep
#             diff[idx_inv[changing_position[j]]] = delta[j]
#             y[:reduced_dim] = v_keep / (diff)
#             y_norm = np.linalg.norm(y)
#             if y_norm > 1e-15:
#                 y /= y_norm

#             y = P.T @ y
#             vec = np.concatenate((eigvecs_left @ y[:n1], eigvecs_right @ y[n1:]))
#             vec /= np.linalg.norm(vec)
#             eigenpairs.append((lam[j], vec))

#         for eigval, vec in zip(deflated_eigvals, deflated_eigvecs):
#             # vec = Q_block @ vec
#             vec = np.concatenate((eigvecs_left @ vec[:n1], eigvecs_right @ vec[n1:]))
#             eigenpairs.append((eigval, vec))
#         eigenpairs.sort(key=lambda x: x[0])
#         final_eigvals = np.array([ev for ev, _ in eigenpairs])
#         final_eigvecs = np.column_stack([vec for _, vec in eigenpairs])
#     else:
#         final_eigvals, final_eigvecs = None, None

#     final_eigvals = comm.bcast(final_eigvals, root=0)
#     final_eigvecs = comm.bcast(final_eigvecs, root=0)

#     return final_eigvals, final_eigvecs


def parallel_eigen(
    main_diag, off_diag, tol_QR=1e-15, max_iterQR=5000, tol_deflation=1e-15
):
    comm = MPI.COMM_WORLD
    main_diag = comm.bcast(main_diag, root=0)
    off_diag = comm.bcast(off_diag, root=0)
    eigvals, eigvecs = parallel_tridiag_eigen(
        main_diag,
        off_diag,
        comm=comm,
        min_size=1,
        tol_factor=tol_deflation,
        tol_QR=tol_QR,
        max_iterQR=max_iterQR,
    )
    return eigvals, eigvecs


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n = 900
    if rank == 0:

        # import debugpy
        # port = 5678 + rank  # 5678 for rank 0, 5679 for rank 1
        # debugpy.listen(("localhost", port))
        # print(f"Rank {rank} waiting for debugger attach on port {port}")
        # debugpy.wait_for_client()
        # np.random.seed(42)
        main_diag = np.ones(n, dtype=np.float64) * 2.0
        off_diag = np.ones(n - 1, dtype=np.float64) * 1.0
        # eig = np.arange(1, n + 1)
        # A = np.diag(eig)
        # U = scipy.stats.ortho_group.rvs(n)

        # A = U @ A @ U.T
        # A = make_symmetric(A)
        # Lanc = EigenSolver(A)
        # initial_guess = np.random.rand(A.shape[0])
        # if initial_guess[0] == 0:
        #     initial_guess[0] += 1
        # _, main_diag, off_diag = Lanczos_PRO(A=A, q=initial_guess)

        T = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        eig_numpy, eig_vec_numpy = np.linalg.eigh(T)
        # print(eig_numpy)

    else:
        main_diag, off_diag = None, None

    main_diag = comm.bcast(main_diag, root=0)
    off_diag = comm.bcast(off_diag, root=0)

    t_s = time()
    eigvals, eigvecs = parallel_tridiag_eigen(
        main_diag, off_diag, comm=comm, min_size=1, tol_factor=1e-8, tol_QR=1e-14
    )
    t_e = time()
    profile_filename = f"profile_rank{comm.Get_rank()}.lprof"
    with open(profile_filename, "w") as f:
        profile.print_stats(stream=f)
    print(f"Elapsed {t_e-t_s}")
    print("Ended the calculation")
    if rank == 0:
        print("I'm in the last part")

        index = np.argsort(eig_numpy)
        eig_numpy = eig_numpy[index]
        eig_vec_numpy[:, index]
        index = np.argsort(eigvals)
        eigvals = eigvals[index]
        eigvecs = eigvecs[:, index]
        # print("Eigenvalue divide and conquer", eigvals)
        # print("Eigenvalue numpy", eig_numpy)

        # print("\n\n Errror eigen", eig_numpy - eigvals)

        print("Norm difference eigenaval", np.linalg.norm(eig_numpy - eigvals, np.inf))

        check_column_directions(eigvecs, eig_vec_numpy)

        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        # with open("eigenvectors_output.txt", "w") as f:
        #     print("Eigenvector solver:\n", eigvecs, file=f)
        #     print("Eigenvector numpy:\n", eig_vec_numpy, file=f)
        #     print("\n\n\nDifference:\n", eig_vec_numpy - eigvecs, file=f)

        # # print("Eigenvector solver:\n", eigvecs)
        # # print("Eigenvector numpy:\n", eig_vec_numpy)
        # print("\n\n\nDifference :\n", eig_vec_numpy - eigvecs)
        diff = eig_vec_numpy - eigvecs
        flat_idx = np.argmax(diff)  # → 5   (counting row-major: 0..8)

        # # If you want row/column coordinates instead of the flattened index:
        row, col = np.unravel_index(flat_idx, diff.shape)  # → (1, 2)
        print(np.max(np.abs(diff)))
        # print("\n\n", eig_vec_numpy[:, col], eigvecs[:, col])
        # print("Norm difference eigenvec", np.linalg.norm(eig_vec_numpy-eigvecs, np.inf))
        """
        D = np.diag(range(n))
        v = np.ones_like(np.diag(D))
        beta = 1e-2
        _ = deflate_eigenpairs(D, v, beta)
        _ = deflate_eigenpairs(D, v, beta)
        beginning = time.time()
        _ = deflate_eigenpairs(D,v,beta)
        end = time.time()
        """
    # n=1000
    # main_diag = np.random.rand(n)
    #  = np.random.rand(n - 1)
    # T = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    # parallel_eigen(main_diag, off_diag)off_diag
