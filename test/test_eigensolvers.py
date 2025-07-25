import numpy as np
import scipy.sparse as sp

import scipy
import pytest
from pyclassify import (
    eigenvalues_np,
    eigenvalues_sp,
    power_method,
    power_method_numba,
    Lanczos_PRO,
    EigenSolver,
)
from pyclassify.utils import make_symmetric


@pytest.fixture(autouse=True)
def set_random_seed():
    seed = 1422
    np.random.seed(seed)


sizes = [20, 50, 100]
densities = [0.05, 0.1, 0.2]


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("density", densities)
def test_implementations_power_method(size, density):
    matrix = sp.random(size, size, density=density, format="csr")
    matrix = make_symmetric(matrix)

    eigs_np = eigenvalues_np(matrix.toarray(), symmetric=True)
    eigs_sp = eigenvalues_sp(matrix, symmetric=True)

    index_np = np.argmax(np.abs(eigs_np))
    index_sp = np.argmax(np.abs(eigs_sp))

    biggest_eigenvalue_np = eigs_np[index_np]
    biggest_eigenvalue_sp = eigs_sp[index_sp]

    biggest_eigenvalue_pm = power_method(matrix)
    biggest_eigenvalue_pm_numba = power_method_numba(matrix.toarray())

    assert np.isclose(
        biggest_eigenvalue_np, biggest_eigenvalue_sp, rtol=1e-5
    )  # ensure numpy and scipy implementations are consistent
    assert np.isclose(
        biggest_eigenvalue_pm, biggest_eigenvalue_sp, rtol=1e-5
    )  # ensure power method and scipy implementation are consistent
    assert np.isclose(
        biggest_eigenvalue_pm_numba, biggest_eigenvalue_sp, rtol=1e-5
    )  # ensure numba power method and scipy implementation are consistent


@pytest.mark.parametrize("size", sizes)
def test_Lanczos(size):
    eig = np.arange(1, size + 1)
    A = np.diag(eig)
    U = scipy.stats.ortho_group.rvs(size)

    A = U @ A @ U.T
    A = make_symmetric(A)

    initial_guess = np.random.rand(A.shape[0])
    if initial_guess[0] == 0:
        initial_guess[0] += 1
    Q, alpha, beta = Lanczos_PRO(A=A, q=initial_guess)

    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)

    EigVal_T = np.linalg.eig(T)[0]
    EigVect_T = np.linalg.eig(T)[1]

    EigVal_A = np.linalg.eig(A)[0]
    EigVect_A = np.linalg.eig(A)[1]

    EigVal_A = np.sort(EigVal_A)
    EigVal_T = np.sort(EigVal_T)

    assert np.allclose(EigVal_T, EigVal_A, rtol=1e-6)

    with pytest.raises(ValueError):
        random_matrix = np.random.rand(size, 2 * size)
        _ = Lanczos_PRO(A=random_matrix, q=initial_guess)

    with pytest.raises(ValueError):
        random_matrix = np.random.rand(size, size)
        _ = Lanczos_PRO(A=random_matrix, q=np.random.rand(2 * size))


@pytest.mark.parametrize("size", sizes)
def test_EigenSolver(size):
    eig = np.arange(1, size + 1)
    A = np.diag(eig)
    U = scipy.stats.ortho_group.rvs(size)

    A = U @ A @ U.T
    A = make_symmetric(A)

    eigensolver = EigenSolver(A, max_iter=int(100 * size), tol=1e-9)

    eigs_np = np.linalg.eig(A)
    eigs_np = np.sort(eig)
    # eig_vec = np.linalg.eig(A).eigenvectors
    # eig_vec = eig_vec / np.linalg.norm(eig_vec, axis=0)

    eigs_QR, _ = eigensolver.eig()
    eigs_QR = np.sort(eigs_QR)
    assert np.allclose(eigs_np, eigs_QR, rtol=1e-4)

    with pytest.raises(ValueError):
        _ = eigensolver.compute_eigenval(diag=np.arange(2), off_diag=np.arange(49))


# @pytest.mark.parametrize("size", sizes)
# @pytest.mark.parametrize("density", densities)
# def test_cupy(size, density):
#    try:
#        if not cp.cuda.is_available():
#            pytest.skip("Skipping test because CUDA is not available")
#    except cp.cuda.runtime.CUDARuntimeError as e:
#        pytest.skip(f"Skipping test due to CUDA driver issues: {str(e)}")
#
#    cp.random.seed(8422)
#
#    matrix = sp.random(size, size, density=density, format="csr")
#    symmetric_matrix = make_symmetric(matrix)
#    cp_symm_matrix = cpsp.csr_matrix(symmetric_matrix)
#    check_square_matrix(cp_symm_matrix)
#
#    eigs_cp = eigenvalues_cp(cp_symm_matrix)
#
#    index_cp = cp.argmax(cp.abs(eigs_cp))
#    biggest_eigenvalue_cp = eigs_cp[index_cp]
#    biggest_eigenvalue_pm_cp = power_method_cp(cp_symm_matrix, max_iter=1000, tol=1e-5)
#
#    assert np.isclose(
#        biggest_eigenvalue_cp, biggest_eigenvalue_pm_cp, rtol=1e-4
#    )  # ensure cupy native and cupy power method implementations are consistent
#
#    random_vector = cp.random.rand(size)
#    eigs_QR, _ = QR_cp(cp_symm_matrix, random_vector, tol=1e-4, max_iter=20 * size)
#    index_cp_QR = cp.argmax(cp.abs(eigs_QR))
#    biggest_eigenvalue_QR = eigs_QR[index_cp_QR]
#    assert cp.isclose(biggest_eigenvalue_cp, biggest_eigenvalue_QR, rtol=1e-3)
