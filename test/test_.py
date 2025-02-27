import numpy as np
import scipy.sparse as sp
import pytest
from pyclassify.utils import eigenvalues_np, eigenvalues_sp, power_method, make_symmetric


np.random.seed(42)
sizes = [10, 100, 1000]
densities = [0.1, 0.5]



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
def test_power_method(size, density):
    matrix = sp.random(size, size, density=density, format="csr")
    matrix = make_symmetric(matrix)

    eigs_np = eigenvalues_np(matrix.toarray(), symmetric=True)
    eigs_sp = eigenvalues_sp(matrix, symmetric=True)

    index_np = np.argmax(np.abs(eigs_np))
    index_sp = np.argmax(np.abs(eigs_sp))

    biggest_eigenvalue_np = eigs_np[index_np]
    biggest_eigenvalue_sp = eigs_sp[index_sp]

    biggest_eigenvalue_pm = power_method(matrix)

    assert np.isclose(biggest_eigenvalue_np, biggest_eigenvalue_sp, rtol=1e-4)
    assert np.isclose(biggest_eigenvalue_pm, biggest_eigenvalue_sp, rtol=1e-4)

    ugly_nonsquare_matrix = np.random.rand(5, 3)

    with pytest.raises(ValueError):
        _ = eigenvalues_np(eigenvalues_np(ugly_nonsquare_matrix, symmetric=True))

    with pytest.raises(ValueError):
        _ = eigenvalues_sp(eigenvalues_sp(ugly_nonsquare_matrix, symmetric=True))