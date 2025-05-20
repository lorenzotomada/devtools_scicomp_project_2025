import numpy as np
from functools import partial


def inner_outer_eigs(eigs, rho):
    inner_eigs = eigs[:-1] if rho > 0 else eigs[1:]
    outer_eig = eigs[-1] if rho > 0 else eigs[0]
    return inner_eigs, outer_eig


def return_secular_f(rho, d, v):
    """
    This returns f as a callable object (function of lambda). f is built using rho, d, v.
    This function passes the tests in test.py and is likely implemented correctly.
    """

    def f(lambda_):
        v_squared = v * v
        return 1 - rho * np.sum(
            [v_squared[k] / (lambda_ - d[k]) for k in range(len(d))]
        )

    return f


def secular_function(mu, rho, d, v2, i):
    """
    Needed to compute the zeros of the secular function in the i-th subinterval;
    """
    psi1, _, psi2, _ = compute_psi_s(mu, rho, d, v2, i)
    return 1 + psi1 + psi2


def check_is_root(f, x, tol=1e-7):
    """
    Usually the values of f(found_eigenvalue) are around 1e-10, so we cannot be *too* restrictive with the threshold.
    For instance, using np.isclose, sometimes the checks are not passed, even though f(found_eig) is very small in
    absolute value and we are indeed very close to one.
    That is the reason for defining a helper function.
    """
    return np.abs(f(x)) < tol


def bisection(f, a, b, tol, max_iter):
    """
    In the main method, we used a slightly tweaked form of bisection for the eig. in the outer interval (i.e. lambda_0 if rho < 0,
    else lambda_{n-1}.
    This helper function implements standard bisection and it is used in the function compute_outer_zero.
    """
    iter_count = 0

    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if np.abs(f(c)) < tol:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1
        if iter_count >= max_iter:
            break
    return (a + b) / 2


def compute_outer_zero(f, rho, interval_end, v, tol=1e-12, max_iter=2000):
    """
    Function to compute the outer eigenvalue (lambda[0] if rho < 0, else lambda[n-1]).
    What it does is the following:
    1) depending on rho, understand whether we should look for it in (d[n-1],+ \infty) or (-\infty, d[0])
    2) use bisection as follows:
      2a) Fix the bisection interval. Notice that (assuming rho > 0, else it is equivalent but specular)
          f(d[n-1]+\epsilon)\approx -\infty, in particular f(d[n-1]+\epsilon)<0.
          So we just need to find r\in\mathbb{R} such that f(d[n-1]+r)>0, and we can regularly use bisection.
      2b) To find that value of r, we just add arbitrary values to d[n-1] until the condition is satisfied.
      2c) Bisection is then used

    Notice that this function passes all the tests, and I assume it is implemented correctly.
    """
    threshold = 1e-11
    update = np.linalg.norm(v)
    if rho > 0:  # f is increasing
        a = interval_end + threshold
        b = interval_end + 1
        while np.sign(f(a) * f(b)) > 0:
            a = b  # if f(b) is still of the same sign, we set a = b, restricting the window for bisection
            b += update  # at some point, f(b) will have a different sign
    elif rho < 0:
        b = interval_end - threshold
        a = interval_end - 1
        while np.sign(f(a) * f(b)) > 0:
            b = a
            a -= update
    x = bisection(f, a, b, tol, max_iter)
    return x


def compute_psi_s(lambda_guess, rho, d, v_squared, i):
    """
    This function computes psi1, psi2 and their derivatives.
    Corresponding to the interval (d[i], d[i+1]).
    Unless I made some mistake, in case rho!=0, it should be sufficient to multiply all the sums by rho, which is what
    is done here and seems to work in case rho > 0.
    """
    denom1 = d[: i + 1] - lambda_guess
    denom2 = d[i + 1 :] - lambda_guess
    psi1 = rho * np.sum(v_squared[: i + 1] / denom1)
    psi1s = rho * np.sum(v_squared[: i + 1] / denom1**2)
    psi2 = rho * np.sum(v_squared[i + 1 :] / denom2)
    psi2s = rho * np.sum(v_squared[i + 1 :] / denom2**2)
    return psi1, psi1s, psi2, psi2s


def compute_inner_zero(rho, d, v, i, tol=1e-12, max_iter=1000):
    """
    This function (or some of its dependencies) surely contains a bug.
    """

    # fix the correct interval
    di = d[i]
    di1 = d[i + 1]

    threshold = 1e-6

    lambda_mid = (di + di1) / 2

    f = return_secular_f(rho, d, v)
    eta = 1.0

    # Decide shift direction based on f at midpoint
    if f(lambda_mid) * rho >= 0:
        shift = di  # the root is closer to di
    else:
        shift = di1  # else, in the right part of the interval

    delta = d - shift  # shift all the quantities, as in the provided code
    delta_i = delta[i]
    delta_i1 = delta[i + 1]

    v2 = (
        v * v
    )  # square once to avoid repeating. This is a bug in the original MATLAB code.

    f_shifted = partial(secular_function, rho=rho, d=delta, v2=v2, i=i)

    mu = bisection(
        f_shifted, delta_i + threshold, delta_i1 - threshold, tol=tol, max_iter=max_iter
    )

    eig = mu + shift
    return eig


def compute_eigenvalues(rho, d, v):
    f = return_secular_f(rho, d, v)
    eigenvalues = []
    iter_range = range(len(d) - 1)

    if rho < 0:
        min_eigenvalue = compute_outer_zero(f, rho, d[0], v)
        eigenvalues.append(min_eigenvalue)
    elif rho > 0:
        max_eigenvalue = compute_outer_zero(f, rho, d[-1], v)
        eigenvalues.append(max_eigenvalue)

    for i in iter_range:  # this can be parallelized
        eig = compute_inner_zero(rho, d, v, i)
        eigenvalues.append(eig)

    return np.sort(np.array(eigenvalues))
