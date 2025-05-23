import numpy as np
from functools import partial


def inner_outer_eigs(eigs, rho):
    """
    Splits the eigenvalues into inner eigenvalues and the outer eigenvalue based on the sign of rho.

    Parameters:
    eigs (np.ndarray or scipy.sparse.spmatrix): Array of eigenvalues, assumed to be sorted.
    rho (float): Scalar parameter appearing in the secular function (please refer to the documentation for more detailed info).

    Returns:
    tuple: A tuple (inner_eigs, outer_eig) where inner_eigs is an array of eigenvalues and outer_eig is a scalar.
           If rho > 0, the last eigenvalue is considered outer due to the interlacing property; otherwise, the first is.
    """
    inner_eigs = eigs[:-1] if rho > 0 else eigs[1:]
    outer_eig = eigs[-1] if rho > 0 else eigs[0]
    return inner_eigs, outer_eig


def return_secular_f(rho, d, v):
    """
    Constructs the secular function for a rank-one update to a diagonal matrix.

    Parameters:
    rho (float): Scalar from the rank-one matrix update.
    d (np.ndarray or scipy.sparse.spmatrix): 1D array or sparse vector of diagonal entries.
    v (np.ndarray or scipy.sparse.spmatrix): 1D array or sparse vector used in the rank-one update.

    Returns:
    callable:
        f(lambda_: float) -> float: The secular function evaluated at lambda_.
    """

    def f(lambda_):
        v_squared = v * v
        return 1 - rho * np.sum(
            [v_squared[k] / (lambda_ - d[k]) for k in range(len(d))]
        )

    return f


def secular_function(mu, rho, d, v2, i):
    """
    Evaluates the secular function at a given point in the i-th subinterval.

    Parameters:
    mu (float): Point at which to evaluate the secular function.
    rho (float): Scalar from the rank-one matrix update.
    d (np.ndarray or scipy.sparse.spmatrix): 1D array or sparse vector of diagonal entries.
    v2 (np.ndarray or scipy.sparse.spmatrix): Elementwise square of the update vector v (i.e., v ** 2).
    i (int): Index of the subinterval in which mu lies.

    Returns:
    float: The value of the secular function at mu.
    """
    psi1, _, psi2, _ = compute_psi_s(mu, rho, d, v2, i)
    return 1 + psi1 + psi2


def check_is_root(f, x, tol=1e-7):
    """
    Determines whether x is a root of the function f within a given numerical tolerance.
    Written because np.isclose is too restrictive even in cases in which we are indeed close to an eigenvalue.

    Parameters:
    f (callable): Function to evaluate.
    x (float): Point to test as a root.
    tol (float): Absolute tolerance for considering f(x) close to zero.

    Returns:
    bool: True if |f(x)| < tol, indicating x is a root within the specified tolerance.
    """
    return np.abs(f(x)) < tol


def bisection(f, a, b, tol, max_iter):
    """
    Standard bisection method to find a root of the function f in the interval [a, b].

    This implementation is used in `compute_outer_zero` to locate the outer eigenvalue.

    Parameters:
    f (callable): A continuous function for which f(a) * f(b) < 0.
    a (float): Left endpoint of the interval.
    b (float): Right endpoint of the interval.
    tol (float): Tolerance for convergence. The method stops when the interval is smaller than tol or when f(c) is sufficiently small.
    max_iter (int): Maximum number of iterations before stopping.

    Returns:
    float: Approximation of the root within the specified tolerance.
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
    Computes the outer eigenvalue (lambda[0] if rho < 0, lambda[n-1] if rho > 0) of a rank-one modified diagonal matrix.

    The secular function  behaves such that:
      - If rho > 0, the outer eigenvalue lies in (d[n-1], infty), and f is increasing in this interval.
      - If rho < 0, the outer eigenvalue lies in (-infty, d[0]), and f is decreasing in this interval.

    This function:
    1. Determines the direction to search based on the sign of rho.
    2. Finds an upper bound (or lower bound for rho < 0) where the secular function changes sign.
    3. Uses the bisection method to find the root in the determined interval.

    Parameters:
    f (callable): The secular function to find a root of.
    rho (float): Scalar rank-one update parameter.
    interval_end (float): Either d[0] or d[n-1], depending on the sign of rho.
    v (np.ndarray or scipy.sparse.spmatrix): Vector from the rank-one update; used to scale the search step size.
    tol (float, optional): Convergence tolerance. Default is 1e-12.
    max_iter (int, optional): Maximum number of bisection iterations. Default is 2000.

    Returns:
    float: Approximation of the outer eigenvalue.
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
    Computes partial sums (psi1, psi2) and their derivatives for the secular function
    in the i-th interval (d[i], d[i+1]).

    Parameters:
    lambda_guess (float): Evaluation point for the secular function.
    rho (float): Scalar rank-one update parameter.
    d (np.ndarray or scipy.sparse.spmatrix): 1D array of diagonal entries.
    v_squared (np.ndarray or scipy.sparse.spmatrix): Precomputed elementwise square of the update vector v.
    i (int): Index defining the interval (d[i], d[i+1]).

    Returns:
    tuple: (psi1, psi1', psi2, psi2') — the partial secular sums and their derivatives.
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
    Computes the i-th eigenvalue that lies in the interval (d[i], d[i+1]) for a
    rank-one modified diagonal matrix using the secular equation.

    Parameters:
    rho (float): Rank-one update scalar.
    d (np.ndarray or scipy.sparse.spmatrix): 1D array of diagonal entries, sorted in ascending order.
    v (np.ndarray or scipy.sparse.spmatrix): 1D update vector.
    i (int): Index indicating the interval (d[i], d[i+1]) to find the zero in.
    tol (float, optional): Tolerance for root-finding. Default is 1e-12.
    max_iter (int, optional): Maximum iterations for bisection. Default is 1000.

    Returns:
    float: The computed inner eigenvalue in the interval (d[i], d[i+1]).
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
    """
    Computes all eigenvalues of a rank-one modified diagonal matrix D + rho * v v^T
    using the secular equation method.

    Parameters:
    rho (float): Rank-one update scalar.
    d (np.ndarray or scipy.sparse.spmatrix): 1D array of sorted diagonal entries of D.
    v (np.ndarray or scipy.sparse.spmatrix): Update vector v in the rank-one perturbation.

    Returns:
    np.ndarray: Sorted array of all eigenvalues of the perturbed matrix.
    """
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
