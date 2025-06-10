import numpy as np
import scipy.sparse as sp

import scipy
import pytest
from pyclassify import eigenvalues_np, eigenvalues_sp
from pyclassify.zero_finder import compute_Psi, secular_solver


@pytest.fixture(autouse=True)
def set_random_seed():
    seed = 1422
    np.random.seed(seed)


d_s = [np.array([-0.3, -0.2, 0.1, 0.2, 0.3]), np.array([-0.1, 0.1, 0.2, 0.3, 0.4])]
rho_s = [1.5, -2]
v_s = [np.array([0.3, 0.2, 0.24, 0.34, 1]), np.array([-1, 0.3, 2, 0.4, 2])]
i_s = [0, 1, 2, 3]


@pytest.mark.parametrize("d", d_s)
@pytest.mark.parametrize("rho", rho_s)
@pytest.mark.parametrize("v", v_s)
@pytest.mark.parametrize("i", i_s)
def test_psi_s(rho, d, v, i):
    """
    Test to check that the psi_s appearing in the nonlinear solver satisfy the theoretical constraint.
    """
    v = v**2
    lambda_guess = (d[i + 1] + d[i]) / 2
    psi1, psi2, psi1_prime, psi2_prime = compute_Psi(i, v, d, rho)

    assert (
        psi1_prime(lambda_guess) * psi2_prime(lambda_guess) >= 0
    ), "Error. The sign of psi1s or psi2s is wrong."
    assert psi1(lambda_guess) * rho <= 0, "Error. Inconsistent with the theory."
    assert psi2(lambda_guess) * rho >= 0, "Error. Inconsistent with the theory."


@pytest.mark.parametrize("d", d_s)
@pytest.mark.parametrize("rho", rho_s)
@pytest.mark.parametrize("v", v_s)
def test_compute_eigenvalues(rho, d, v):
    D = np.diag(d)
    rk_1_update = rho * np.outer(v, v)
    L = D + rk_1_update
    computed_eigs, _, __ = secular_solver(rho, d, v)
    print(len(computed_eigs))
    print(computed_eigs[1])
    print(computed_eigs)
    computed_eigs = np.sort(computed_eigs)
    eigs, _ = np.linalg.eig(L)

    exact_eigs, _ = np.linalg.eig(L)
    exact_eigs = np.sort(exact_eigs)

    for i in range(len(exact_eigs)):
        assert (
            np.abs(computed_eigs[i] - exact_eigs[i]) < 1e-8
        ), "Error. The eigenvalues were not computed correctly."


@pytest.mark.parametrize("d", d_s)
@pytest.mark.parametrize("rho", rho_s)
@pytest.mark.parametrize("v", v_s)
def test_compute_eigenvalues(rho, d, v):
    D = np.diag(d)
    rk_1_update = rho * np.outer(v, v)
    L = D + rk_1_update
    computed_eigs, _, __ = secular_solver(rho, d, v)
    print(len(computed_eigs))
    print(computed_eigs[1])
    print(computed_eigs)
    computed_eigs = np.sort(computed_eigs)
    eigs, _ = np.linalg.eig(L)

    exact_eigs, _ = np.linalg.eig(L)
    exact_eigs = np.sort(exact_eigs)

    for i in range(len(exact_eigs)):
        assert (
            np.abs(computed_eigs[i] - exact_eigs[i]) < 1e-8
        ), "Error. The eigenvalues were not computed correctly."
