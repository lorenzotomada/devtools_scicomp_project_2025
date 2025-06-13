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


def out_range(v, d, rho, lam_0, tol=1e-15, maxiter=100):
    """
    Computes the root of the secular equation outside the range [min(d)=d[0], max(d)=d[-1]].
    Inputs:
        -v: Rank one correction
        -d: Diagonal element of the Diagonal matrix described in equation 5.7
        -rho: off diagonal element used for the splitting.
    Outputs:
        -shift + lam_0: i-th smallest eigenvalue that
    """

    diag = d.copy()
    if rho < 0:
        diag = diag - d[0]
        lam_0 = lam_0 - d[0]
        shift = d[0]
        d_i = diag[0]
        # Psi_1, Psi_2, dPsi_1, dPsi_2 = compute_Psi(-1, v, diag, rho)

        # for _ in range(maxiter):
        #     c_1=dPsi_2(lam_0)*(diag[0]-lam_0)**2
        #     c_3=Psi_2(lam_0) - dPsi_2(lam_0)*(diag[0]-lam_0)+1
        #     lam=diag[0]+c_1/c_3
        #     if abs(lam_0-lam) < tol * max(1.0, abs(lam)):
        #         break
        #     lam_0=lam
        # return shift + lam_0
    else:
        diag = diag - d[-1]
        lam_0 = lam_0 - d[-1]
        shift = d[-1]
        d_i = diag[-1]
    Psi_1, Psi_2, dPsi_1, dPsi_2 = compute_Psi(-1, v, diag, rho)
    for _ in range(maxiter):
        c_1 = dPsi_2(lam_0) * (d_i - lam_0) ** 2
        c_3 = Psi_2(lam_0) - dPsi_2(lam_0) * (d_i - lam_0) + 1
        lam = d_i + c_1 / c_3
        if abs(lam_0 - lam) < tol * max(1e-6, abs(lam)):
            break
        lam_0 = lam
    return shift + lam_0


def secular_solver(rho, d, v):
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
        lam_0 = d[-1] + 10 * (d[-1] - d[-2])
        Eig = out_range(v, d, rho, lam_0)
        eig_val.append(Eig)
        index.append(len(d) - 1)
        delta.append(Eig - d[-1])

    else:
        i = 0
        lam_0 = d[0] - 5 * (d[1] - d[0])
        left_center = False
        index.append(0)
        Eig = out_range(v, d, rho, lam_0)
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
    secular_eigvals = secular_solver(rho, d, v)
    secular_eigvals = np.array(secular_eigvals)
    secular_eigvals.sort()
    print("Secular equation zeros (eigenvalues):", secular_eigvals)
    print("Absolute error:", np.abs(secular_eigvals - direct_eigvals))
