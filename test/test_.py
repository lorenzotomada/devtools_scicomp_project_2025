import numpy as np
import scipy.sparse as sp
import pytest
from pyclassify import (
    eigenvalues_np,
    eigenvalues_sp,
    power_method,
    power_method_numba,
)
from pyclassify.utils import check_A_square_matrix, make_symmetric


np.random.seed(42)


def test_checks_on_A():
    ugly_nonsquare_matrix = np.random.rand(5, 3)

    with pytest.raises(ValueError):
        check_A_square_matrix(ugly_nonsquare_matrix)
    with pytest.raises(TypeError):
        check_A_square_matrix("definitely_not_a_matrix")


sizes = [10, 100, 1000]
densities = [0.1, 0.3]


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("density", densities)
def test_make_symmetric(size, density):
    matrix = sp.random(size, size, density=density, format="csr")
    symmetric_matrix = make_symmetric(matrix)

    assert np.allclose(symmetric_matrix.toarray(), symmetric_matrix.toarray().T)
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

    assert np.isclose(biggest_eigenvalue_np, biggest_eigenvalue_sp, rtol=1e-4)
    assert np.isclose(biggest_eigenvalue_pm, biggest_eigenvalue_sp, rtol=1e-4)
    assert np.isclose(biggest_eigenvalue_pm_numba, biggest_eigenvalue_sp, rtol=1e-4)
