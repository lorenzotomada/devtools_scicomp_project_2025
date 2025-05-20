import pytest
import numpy as np
from pyclassify.helpers_secular import (
    return_secular_f,
    check_is_root,
    compute_inner_zero,
    compute_outer_zero,
    compute_psi_s,
    compute_eigenvalues,
    inner_outer_eigs,
)


d_s = [np.array([-0.3, -0.2, 0.1, 0.2, 0.3]), np.array([-0.1, 0.1, 0.2, 0.3, 0.4])]


rho_s = [1.5, -2]


v_s = [np.array([0.3, 0.2, 0.24, 0.34, 1]), np.array([-1, 0.3, 2, 0.4, 2])]


i_s = [1, 2, 3, 4]


@pytest.mark.parametrize("d", d_s)
@pytest.mark.parametrize("rho", rho_s)
@pytest.mark.parametrize("v", v_s)
def test_f_implementation(rho, d, v):
    """
    Test to check that the f defining the secular equation is implemented correctly.
    """
    D = np.diag(d)
    rk_1_update = rho * np.outer(v, v)
    assert rk_1_update.shape == D.shape, "Error. Shape mismatch."
    L = D + rk_1_update
    f = return_secular_f(rho, d, v)
    for d_i in d:
        with np.errstate(divide="ignore", invalid="ignore"):
            val = f(d_i)
            assert np.isinf(val) or np.isnan(
                val
            ), "f(d_i) should be ±inf or NaN due to division by zero."
    eigs, _ = np.linalg.eig(L)
    for eig in eigs:
        assert check_is_root(f, eig)


@pytest.mark.parametrize("d", d_s)
@pytest.mark.parametrize("rho", rho_s)
@pytest.mark.parametrize("v", v_s)
@pytest.mark.parametrize("i", i_s)
def test_psi_s(rho, d, v, i):
    """
    Test to check that the psi_s appearing in the nonlinear solver satisfy the theoretical constraint.
    """
    v = v**2

    if i == d.shape[0] - 1:
        with pytest.raises(IndexError):
            d_shifted = d
            lambda_guess = (d_shifted[i + 1] - d_shifted[i]) / 2
            psi1, psi1s, psi2, psi2s = compute_psi_s(lambda_guess, rho, d_shifted, v, i)
    else:
        d_shifted = d
        lambda_guess = (d_shifted[i + 1] + d_shifted[i]) / 2
        print(lambda_guess)
        print(d_shifted[i])
        print(d_shifted[i + 1])
        psi1, psi1s, psi2, psi2s = compute_psi_s(lambda_guess, rho, d_shifted, v, i)
        assert psi1s * psi2s >= 0, "Error. The sign of psi1s or psi2s is wrong."
        assert psi1 * rho <= 0
        assert psi2 * rho >= 0
        assert (
            rho * (psi1 - psi1s * (d_shifted[i] - lambda_guess)) <= 0
        ), "Error. Inconsistent with the theory."
        assert (
            rho * (psi2 - psi2s * (d_shifted[i] - lambda_guess)) >= 0
        ), "Error. Inconsistent with the theory."


@pytest.mark.parametrize("d", d_s)
@pytest.mark.parametrize("rho", rho_s)
@pytest.mark.parametrize("v", v_s)
def test_outer_eigenvalues(rho, d, v):
    """
    Test to check that the solver correctly computes the outer eigenvalue.
    """
    f = return_secular_f(rho, d, v)
    interval_end = d[0] if rho < 0 else d[-1]
    eig = compute_outer_zero(f, rho, interval_end, v)
    exact_eigs, _ = np.linalg.eig(np.diag(d) + rho * np.outer(v, v))
    extreme_eigenval = np.min(exact_eigs) if rho < 0 else np.max(exact_eigs)
    assert check_is_root(f, eig), "The eigenvalue has not been computed correctly."
    assert np.abs(
        extreme_eigenval - eig
    ), "The eigenvalue has not been computed correctly."


@pytest.mark.parametrize("d", d_s)
@pytest.mark.parametrize("rho", rho_s)
@pytest.mark.parametrize("v", v_s)
@pytest.mark.parametrize("i", i_s)
def test_inner_eigenvalues(d, rho, v, i):
    """
    Test to check that the solver correctly computes the inner eigenvalues.
    """
    n = d.shape[0]
    f = return_secular_f(rho, d, v)
    iter_range = range(n - 1)
    for i in iter_range:
        eig = compute_inner_zero(rho, d, v, i)
        assert check_is_root(f, eig)


@pytest.mark.parametrize("d", d_s)
@pytest.mark.parametrize("rho", rho_s)
@pytest.mark.parametrize("v", v_s)
def test_compute_eigenvalues(rho, d, v):
    D = np.diag(d)
    rk_1_update = rho * np.outer(v, v)
    L = D + rk_1_update
    f = return_secular_f(rho, d, v)
    computed_eigs = compute_eigenvalues(rho, d, v)
    eigs, _ = np.linalg.eig(L)

    for eig in computed_eigs:
        assert check_is_root(f, eig)

    exact_eigs, _ = np.linalg.eig(L)
    exact_eigs = np.sort(exact_eigs)
    exact_inner_eigs, exact_outer_eig = inner_outer_eigs(exact_eigs, rho)
    computed_inner_eigs, computed_outer_eig = inner_outer_eigs(computed_eigs, rho)

    assert np.isclose(
        computed_outer_eig, exact_outer_eig
    ), "Error. Outer eigenvalue computed incorrectly"

    print(f"rho: {rho}")
    print(f"computed: {computed_inner_eigs}")
    print(f"exact: {exact_inner_eigs}")
    for i in range(len(exact_inner_eigs)):
        assert np.isclose(
            exact_inner_eigs[i], computed_inner_eigs[i]
        ), "Error. The eigenvalues were not computed correctly."
