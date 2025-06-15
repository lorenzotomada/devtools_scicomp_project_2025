import numpy as np


def compute_Psi(i, v, d, rho):
    """
    This function computes the functions Psi1, and Psi2 and their derivative, accordingly to the equation described
    by the notes chapter 5-6 contained in the shared folders.
    The equation utilized are the equation 5.27, 5.25 and 5.29.
    d is assumed to be already a vector whose elements are already sorted.
    Inputs:
        -i: It is the index of the i-th smallest eigenvalue that is being computed
        -v: Rank one correction
        -d: Diagonal element of the Diagonal matrix described in equation 5.7
        -rho: off diagonal element used for the splitting.
    Output:
        Psi1:  Lamoba function that returns the value of Psi1 at the point x
        Psi2:  Lamoba function that returns the value of Psi2 at the point x
        dPsi1: Lamoba function that returns the value of the derivative dPsi1/dx at the point x
        dPsi2: Lamoba function that returns the value of the derivative dPsi2/dx at the point x
    """
    Psi_1 = lambda x: rho * np.sum(v[: i + 1] ** 2 / (d[: i + 1] - x))
    Psi_2 = lambda x: rho * np.sum(v[i + 1 :] ** 2 / (d[i + 1 :] - x))
    dPsi_1 = lambda x: rho * np.sum(v[: i + 1] ** 2 / (d[: i + 1] - x) ** 2)
    dPsi_2 = lambda x: rho * np.sum(v[i + 1 :] ** 2 / (d[i + 1 :] - x) ** 2)
    return Psi_1, Psi_2, dPsi_1, dPsi_2


def find_root(i, left_center, v, d, rho, lam_0, tol=1e-15, maxiter=100):
    """
    Find the roots of the secular equation contained inside the interval min(d)=d[0] and max(d)=d[-1].
    Inputs:
        -i: It is the index of the i-th smallest eigenvalue that is being computed
        -left_center: boolen variable. True if the left extreme is the center od the new - shifted reference system.
        -v: Rank one correction
        -d: Diagonal element of the Diagonal matrix described in equation 5.7
        -rho: off diagonal element used for the splitting.
        -lam_0: initial guess of the i-th root of the secular equation
        -tol: absolute or relative tollerence. For lamba value below 1e-6, tols refers to the absolute tolerance (avoiding division by zero). For
              lambda value greater than 1e-6, tol is the absolute tolerance.
        -maxiter: Maximum number of iteration of the  iterative solver.

    Outputs:
        -shift + lam_0: i-th smallest eigenvalue
        -lam_0: difference between the i-th eigenvalue and the nearest diagonal element.
    """
    diag = d.copy()
    if left_center:
        diag = diag - d[i]
        lam_0 = lam_0 - d[i]
        shift = d[i]
    else:
        diag = diag - d[i + 1]
        lam_0 = lam_0 - d[i + 1]
        shift = d[i + 1]

    Psi_1, Psi_2, dPsi_1, dPsi_2 = compute_Psi(i, v, diag, rho)

    for _ in range(maxiter):
        delta_i = diag[i] - lam_0
        delta_i1 = diag[i + 1] - lam_0 if i + 1 < len(d) else 0.0
        vPsi_1 = Psi_1(lam_0)
        vPsi_2 = Psi_2(lam_0)
        vdPsi_1 = dPsi_1(lam_0)
        vdPsi_2 = dPsi_2(lam_0)
        a = (1 + vPsi_1 + vPsi_2) * (delta_i + delta_i1) - (
            vdPsi_1 + vdPsi_2
        ) * delta_i * delta_i1
        b = delta_i * delta_i1 * (1 + vPsi_1 + vPsi_2)
        c = 1 + vPsi_1 + vPsi_2 - delta_i * vdPsi_1 - delta_i1 * vdPsi_2
        discr = a**2 - 4 * b * c
        discr = max(discr, 0)

        eta = (a - rho / np.abs(rho) * np.sqrt(discr)) / (2 * c)
        lam_0 += eta
        if abs(eta) < tol * max(1e-8, abs(lam_0)):
            break
    return shift + lam_0, lam_0


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


def compute_outer_zero(v, d, rho, interval_end, tol=1e-12, max_iter=1000):
    """
    Computes the outer eigenvalue (lambda[0] if rho < 0, lambda[n-1] if rho > 0) of a rank-one modified diagonal matrix.

    The secular function  behaves such that:
      - If rho > 0, the outer eigenvalue lies in (d[n-1], infty), and f is increasing in this interval.
      - If rho < 0, the outer eigenvalue lies in (-infty, d[0]), and f is decreasing in this interval.

    This function:
    1. Determines the direction to search based on the sign of rho.
    2. Finds an upper bound (or lower bound for rho < 0) where the secular function changes sign.
    3. Uses the bisection method to find the root in the determined interval.

    Returns:
    float: Approximation of the outer eigenvalue.
    """
    threshold = 1e-11
    update = np.linalg.norm(v)

    f = lambda x: 1 - rho * np.sum([(v * v)[k] / (x - d[k]) for k in range(len(d))])

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


def secular_solver_python(rho, d, v):
    """
    Computes all the roots of the secular equation.
    Inputs:
        -v: Rank one correction
        -d: Diagonal element of the Diagonal matrix described in equation 5.7
        -rho: off diagonal element used for the splitting.
    Outputs:
        -eig_val: vector of the roots of the secular equation
        -index: vector of the index of the center of the shifted frame of reference associated to the i-ith root
        -delta: vector of difference between the i-th eigenvalue and the nearest diagonal element.
    """
    f = lambda x: 1 - rho * np.sum(v**2 / (x - d))
    eig_val = []
    index = []
    delta = []

    if rho > 0:
        for i, d_i in enumerate(d[:-1]):
            lam_0 = (d_i + d[i + 1]) * 0.5
            if f(lam_0) > 0:
                left_center = True
                index.append(i)
            else:
                left_center = False
                index.append(i + 1)
            Eig, Delta = find_root(i, left_center, v, d, rho, lam_0)
            eig_val.append(Eig)
            delta.append(Delta)
        lam_0 = d[-1]
        Eig = compute_outer_zero(v, d, rho, lam_0)
        eig_val.append(Eig)
        index.append(len(d) - 1)
        delta.append(Eig - d[-1])

    else:
        i = 0
        lam_0 = d[0]
        left_center = False
        index.append(0)
        Eig = compute_outer_zero(v, d, rho, lam_0)
        eig_val.append(Eig)
        delta.append(Eig - d[0])

        for i, d_i in enumerate(d[:-1]):
            lam_0 = (d_i + d[i + 1]) * 0.5
            if f(lam_0) > 0:
                left_center = False
                index.append(i + 1)
            else:
                left_center = True
                index.append(i)
            eig_val.append(find_root(i, left_center, v, d, rho, lam_0)[0])
    return (eig_val, index, delta)


if __name__ == "__main__":

    rho = 2
    d = np.array([-0.3, -0.2, 0.1, 0.2, 0.3])
    v = np.array([0.3, 0.2, 0.24, 0.34, 1])
    n = 100
    d = np.random.rand(n)

    v = np.random.rand(n)
    d = np.sort(d)
    T = np.diag(d) + rho * np.outer(v, v)
    direct_eigvals = np.linalg.eigvalsh(np.diag(d) + rho * np.outer(v, v))
    print("Direct eigvals for comparison:", direct_eigvals)
    secular_eigvals, _, __ = secular_solver_python(rho, d, v)
    secular_eigvals = np.array(secular_eigvals)
    secular_eigvals.sort()
    print("Secular equation zeros (eigenvalues):", secular_eigvals)
    print("Absolute error:", np.abs(secular_eigvals - direct_eigvals))
