import numpy as np
import scipy.sparse as sp
import scipy
import pytest
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
