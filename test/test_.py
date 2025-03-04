import numpy as np
import scipy.sparse as sp
import cupyx.scipy.sparse as cpsp
import pytest
from pyclassify import (
    eigenvalues_np,
    eigenvalues_sp,
    eigenvalues_cp,
    power_method,
    power_method_numba,
    power_method_cp,
)
from pyclassify.utils import check_square_matrix, make_symmetric, check_symm_square


np.random.seed(104)


sizes = [10, 100, 1000]
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

    cp_symm_matrix = cpsp.csr_matrix(symmetric_matrix)
    check_square_matrix(cp_symm_matrix)

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
    cp_matrix = cpsp.csr_matrix(matrix)

    eigs_np = eigenvalues_np(matrix.toarray(), symmetric=True)
    eigs_sp = eigenvalues_sp(matrix, symmetric=True)
    eigs_cp = eigenvalues_cp(cp_matrix)

    index_np = np.argmax(np.abs(eigs_np))
    index_sp = np.argmax(np.abs(eigs_sp))
    index_cp = np.argmax(np.abs(eigs_cp))

    biggest_eigenvalue_np = eigs_np[index_np]
    biggest_eigenvalue_sp = eigs_sp[index_sp]
    biggest_eigenvalue_cp = eigs_cp[index_cp]

    biggest_eigenvalue_pm = power_method(matrix)
    biggest_eigenvalue_pm_numba = power_method_numba(matrix.toarray())
    biggest_eigenvalue_pm_cp = power_method_cp(cp_matrix)

    assert np.isclose(
        biggest_eigenvalue_np, biggest_eigenvalue_sp, rtol=1e-4
    )  # ensure numpy and scipy implementations are consistent
    assert np.isclose(
        biggest_eigenvalue_cp, biggest_eigenvalue_sp, rtol=1e-4
    )  # ensure cupy and scipy implementations are consistent
    assert np.isclose(
        biggest_eigenvalue_pm, biggest_eigenvalue_sp, rtol=1e-4
    )  # ensure power method and scipy implementation are consistent
    assert np.isclose(
        biggest_eigenvalue_pm_numba, biggest_eigenvalue_sp, rtol=1e-4
    )  # ensure numba power method and scipy implementation are consistent
    assert np.isclose(
        biggest_eigenvalue_pm_cp, biggest_eigenvalue_sp, rtol=1e-4
    )  # ensure cupy power method and scipy implementation are consistent
