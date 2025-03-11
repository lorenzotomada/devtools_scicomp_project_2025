import numpy as np
import scipy.sparse as sp
import cupyx.scipy.sparse as cpsp
import cupy as cp
import scipy
import pytest
from pyclassify import (
    eigenvalues_np,
    eigenvalues_sp,
    eigenvalues_cp,
    power_method,
    power_method_numba,
    power_method_cp,
    Lanczos_PRO,
    QR_method,
    QR,
    QR_cp,
)
from pyclassify.utils import check_square_matrix, make_symmetric, check_symm_square


@pytest.fixture(autouse=True)
def set_random_seed():
    seed = 1422
    np.random.seed(seed)


sizes = [20, 100]
densities = [0.1, 0.3]


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("density", densities)
def test_checks_on_A(size, density):
    ugly_nonsquare_matrix = np.random.rand(size, size + 1)

    with pytest.raises(ValueError):
        check_square_matrix(ugly_nonsquare_matrix)
    with pytest.raises(TypeError):
        check_square_matrix("definitely_not_a_matrix")

    matrix = sp.random(size, size, density=density, format="csr")
    symmetric_matrix = make_symmetric(matrix)
    check_square_matrix(symmetric_matrix)

    # regarding the CuPy implementation: see below!

    not_so_symmetric_matrix = np.random.rand(5, 5)
    if not_so_symmetric_matrix[1, 2] == not_so_symmetric_matrix[2, 1]:
        not_so_symmetric_matrix[1, 2] += 1
    not_so_symmetric_matrix = sp.csr_matrix(not_so_symmetric_matrix)

    with pytest.raises(ValueError):
        check_symm_square(not_so_symmetric_matrix)


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("density", densities)
def test_make_symmetric(size, density):
    matrix = sp.random(size, size, density=density, format="csr")
    symmetric_matrix = make_symmetric(matrix)

    check_square_matrix(symmetric_matrix)
    assert symmetric_matrix.shape == matrix.shape

    with pytest.raises(TypeError):
        _ = make_symmetric("banana")


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
        biggest_eigenvalue_np, biggest_eigenvalue_sp, rtol=1e-4
    )  # ensure numpy and scipy implementations are consistent
    assert np.isclose(
        biggest_eigenvalue_pm, biggest_eigenvalue_sp, rtol=1e-4
    )  # ensure power method and scipy implementation are consistent
    assert np.isclose(
        biggest_eigenvalue_pm_numba, biggest_eigenvalue_sp, rtol=1e-4
    )  # ensure numba power method and scipy implementation are consistent


@pytest.mark.parametrize("size", sizes)
def test_Lanczos(size):
    A = np.random.rand(size, size)
    A = (A + A.T) / 2
    A = np.array(A, dtype=float)
    q = np.random.rand(size)
    # matrix =np.array(matrix, dtype=np.float64)
    random_vector = np.random.rand(size)

    _, alpha, beta = Lanczos_PRO(A, random_vector)

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
        _ = Lanczos_PRO(random_matrix, random_vector)

    with pytest.raises(ValueError):
        random_matrix = np.random.rand(size, size)
        _ = Lanczos_PRO(random_matrix, np.random.rand(2 * size))


@pytest.mark.parametrize("size", sizes)
def test_QR_method(size):
    eig = np.arange(1, size + 1)
    A = np.diag(eig)
    U = scipy.stats.ortho_group.rvs(size)

    A = U @ A @ U.T
    A = make_symmetric(A)  #  not needed probably
    eig = np.linalg.eig(A)
    index = np.argsort(eig.eigenvalues)
    eig = eig.eigenvalues
    eig_vec = np.linalg.eig(A).eigenvectors
    eig_vec = eig_vec[index]
    eig = eig[index]
    eig_vec = eig_vec / np.linalg.norm(eig_vec, axis=0)

    random_vector = np.random.rand(size)
    _, alpha, beta = Lanczos_PRO(A, random_vector)

    # T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)

    eig_valQR, _ = QR_method(alpha, beta, max_iter=100 * size)
    index = np.argsort(eig_valQR)
    eig_valQR = eig_valQR[index]

    assert np.allclose(eig, eig_valQR, rtol=1e-4)


@pytest.mark.parametrize("size", sizes)
def test_QR(size):
    eig = np.arange(1, size + 1)
    A = np.diag(eig)
    U = scipy.stats.ortho_group.rvs(size)

    A = U @ A @ U.T
    A = make_symmetric(A)
    eig = np.linalg.eig(A)
    index = np.argsort(eig.eigenvalues)
    eig = eig.eigenvalues
    eig_vec = np.linalg.eig(A).eigenvectors
    eig_vec = eig_vec[index]
    eig = eig[index]
    eig_vec = eig_vec / np.linalg.norm(eig_vec, axis=0)

    random_vector = np.random.rand(size)
    eigs_QR, _ = QR(A, random_vector, tol=1e-4, max_iter=1000)

    index = np.argsort(eigs_QR)
    eig_QR = eigs_QR[index]

    assert np.allclose(eig, eig_QR, rtol=1e-4)


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("density", densities)
def test_cupy(size, density):
    try:
        if not cp.cuda.is_available():
            pytest.skip("Skipping test because CUDA is not available")
    except cp.cuda.runtime.CUDARuntimeError as e:
        pytest.skip(f"Skipping test due to CUDA driver issues: {str(e)}")

    cp.random.seed(8422)

    matrix = sp.random(size, size, density=density, format="csr")
    symmetric_matrix = make_symmetric(matrix)
    cp_symm_matrix = cpsp.csr_matrix(symmetric_matrix)
    check_square_matrix(cp_symm_matrix)

    eigs_cp = eigenvalues_cp(cp_symm_matrix)

    index_cp = cp.argmax(cp.abs(eigs_cp))
    biggest_eigenvalue_cp = eigs_cp[index_cp]
    biggest_eigenvalue_pm_cp = power_method_cp(cp_symm_matrix, max_iter=1000, tol=1e-5)

    assert np.isclose(
        biggest_eigenvalue_cp, biggest_eigenvalue_pm_cp, rtol=1e-4
    )  # ensure cupy native and cupy power method implementations are consistent

    random_vector = cp.random.rand(size)
    eigs_QR, _ = QR_cp(cp_symm_matrix, random_vector, tol=1e-4, max_iter=500 * size)
    index_cp_QR = cp.argmax(cp.abs(eigs_QR))
    biggest_eigenvalue_QR = eigs_QR[index_cp_QR]
    assert cp.isclose(biggest_eigenvalue_cp, biggest_eigenvalue_QR, rtol=1e-3)
