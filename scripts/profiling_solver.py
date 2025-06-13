from pyclassify import EigenSolver
from pyclassify.QR_cpp import QR_algorithm
from pyclassify.utils import make_symmetric
import numpy as np
from time import time
import matplotlib.pyplot as plt

# n=10
# np.random.seed(42)
# matrix = np.random.rand(n, n)
# A=make_symmetric(matrix)
# #step 1
# #Check that Lanczos Alogrithm works well

# #creating the object of tha class
# eig_obj=EigenSolver(A)

# #reducing the symmetric matrix to a tridiagonal matrix

# Q, d, off_d=eig_obj.Lanczos_PRO()

# #The matrix A and the trdiagonal should have the same spectrum

# npEig_val, npEig_vec=np.linalg.eigh(A)

# T=np.diag(d) + np.diag(off_d, 1) + np.diag(off_d, -1)
# LaEig_val, LaEig_vec=np.linalg.eigh(T)

# #check if the eigenvalues are the same
# print(f"The maximum of the difference between the eigenvalue of A and the eigenvalue of {np.linalg.norm(npEig_val-LaEig_val, np.inf)}")

# #checking that the unitary matrix Q is correct as well. The decomposition is such that A= Q' T Q (where ' denotes the transpose o)

# print(f"The maximum absoulute error in the decomposition is equal to {np.max(np.abs(A-Q.T@T@Q))}")

# #check if the eigenvalues are the same
# QREig_val, QREig_vec= QR_algorithm(d, off_d)
# index_sort=np.argsort(QREig_val)
# QREig_vec=np.array(QREig_vec)
# QREig_val=np.array(QREig_val)
# QREig_vec=QREig_vec[:, index_sort]
# QREig_val=QREig_val[index_sort]
# print(f"The maximum of the difference between the eigenvalue computed by numpy on A and the developed QR algorithm computed" 
#        f"on the Matrix T is {np.linalg.norm(npEig_val-QREig_val, np.inf)}")



# def check_column_directions(A, B):
#     n = A.shape[1]
#     for i in range(n):
#         # Normalize the vectors
#         a = A[:, i] 
#         b = B[:, i] 
#         dot = np.dot(a, b)
#         # Should be close to 1 or -1
#         if dot<0:
#             B[:, i] = -B[:, i] 
        

# check_column_directions(QREig_vec, LaEig_vec )
# print(f"The maximum absoulute error among the components of all the eigenvectors of T is {np.max(np.abs(QREig_vec - LaEig_vec))}")

# #Profiling the time of Lancsoz algorithm

# i_max=11
# time_vector=np.array([])
# for i in range(3, i_max):
#     n=2**i
#     np.random.seed(42)
#     matrix = np.random.rand(n, n)
#     A=make_symmetric(matrix)
#     q=np.ones(n)
#     t_s=time()
#     eig_obj.Lanczos_PRO(A, q)
#     t_e=time()
#     time_vector=np.append(time_vector, t_e - t_s)

# p=np.polyfit([2**i for i in range(3, i_max)], time_vector, 3)
# print(p)
# plt.plot([2**i for i in range(3, i_max)], time_vector, 'ko')
# x=np.linspace(2**3, 2**(i_max-1))
# plt.plot(x, np.polyval(p, x))
# plt.show()

# n=2**9
# np.random.seed(42)
# matrix = np.random.rand(n, n)
# A=make_symmetric(matrix)
# npEig_val, _=np.linalg.eigh(A)
# q=np.ones(n)
# eps=1e-15
# Q, d, off_d=eig_obj.Lanczos_PRO(A, q, tol=eps)

# eps=[10**(-i) for i in range(2, 15)]
# threshold=1e-6
# tolerance_time_vector=np.array([])
# number_of_nonOrtBase=np.array([])
# error_eigenvalue=np.array([])
# for tol in eps:
#     t_s=time()
#     Q, d, off_d=eig_obj.Lanczos_PRO(A, q, tol=tol)
#     t_e=time()
#     T=np.diag(d) + np.diag(off_d, 1) + np.diag(off_d, -1)
#     LaEig_val, _ =np.linalg.eigh(T)
#     tolerance_time_vector=np.append(tolerance_time_vector, t_e - t_s)
#     error_eigenvalue=np.append(error_eigenvalue, np.linalg.norm(np.abs(npEig_val-LaEig_val), np.inf))
#     #loss of orthogonality
#     G=Q@Q.T
#     G=np.abs(G-np.tril(G))
#     G=G > threshold
#     count = np.sum(G)
#     number_of_nonOrtBase=np.append(number_of_nonOrtBase, count)

# fig, axs = plt.subplots(3, 1, figsize=(7, 12), sharex=True)

# # 1. Number of non-orthogonal base vectors
# axs[0].bar(eps, number_of_nonOrtBase, width=0.8*np.array(eps), align='center')
# axs[0].set_ylabel('Non-orthogonal\nbase vectors')
# axs[0].set_xscale('log')
# axs[0].set_yscale('log')
# axs[0].grid(True, axis='y')
# axs[0].set_title('Lanczos PRO: Convergence & Orthogonality')

# # 2. Time vs Tolerance
# axs[1].plot(eps, tolerance_time_vector, marker='o')
# axs[1].set_ylabel('Time (s)')
# axs[1].set_xscale('log')
# axs[1].grid(True)

# # 3. Error in eigenvalue (log-log)
# axs[2].plot(eps, error_eigenvalue, marker='o')
# axs[2].set_ylabel('Eigenvalue error')
# axs[2].set_xscale('log')
# axs[2].set_yscale('log')
# axs[2].grid(True, which="both")

# #Shared, thick x-axis label only at the bottom
# axs[2].set_xlabel('Tolerance ($\epsilon$)', fontsize=15, fontweight='bold')

# plt.tight_layout()
# plt.show()

# #Time scaling QR

i_max=12

time_vector=np.array([])
error_eigenvalue=np.array([])
for i in range(3, i_max):
    n=2**i
    #np.random.seed(42)
    d = np.random.rand(n)
    off_d=np.random.rand(n-1)
    T=np.diag(d) + np.diag(off_d, 1) + np.diag(off_d, -1)
    npEig_val, _ =np.linalg.eigh(T)
    t_s=time()
    QREig_val, QREig_vec= QR_algorithm(d, off_d)
    t_e=time()
    index_sort=np.argsort(QREig_val)
    QREig_vec=np.array(QREig_vec)
    QREig_val=np.array(QREig_val)
    QREig_vec=QREig_vec[:, index_sort]
    QREig_val=QREig_val[index_sort]
    error_eigenvalue=np.append(error_eigenvalue, np.linalg.norm(np.abs(npEig_val-QREig_val), np.inf))
    time_vector=np.append(time_vector, t_e - t_s)

print(error_eigenvalue)
p=np.polyfit([2**i for i in range(3, i_max)], time_vector, 3)
print(p)
plt.plot([2**i for i in range(3, i_max)], time_vector, 'ko')
x=np.linspace(2**3, 2**(i_max-1))
plt.plot(x, np.polyval(p, x))
plt.show()

# n=2**10
# np.random.seed(42)
# d = np.random.rand(n)
# off_d=np.random.rand(n-1)
# eps=[10**(-i) for i in range(2, 15)]
# T=np.diag(d) + np.diag(off_d, 1) + np.diag(off_d, -1)
# npEig_val, _ =np.linalg.eigh(T)

# tolerance_time_vector=np.array([])
# error_eigenvalue=np.array([])
# for tol in eps:
#     t_s=time()
#     QREig_val, QREig_vec= QR_algorithm(d, off_d, toll=tol)
#     t_e=time()
#     index_sort=np.argsort(QREig_val)
#     QREig_vec=np.array(QREig_vec)
#     QREig_val=np.array(QREig_val)
#     QREig_vec=QREig_vec[:, index_sort]
#     QREig_val=QREig_val[index_sort]
#     tolerance_time_vector=np.append(tolerance_time_vector, t_e - t_s)
#     error_eigenvalue=np.append(error_eigenvalue, np.linalg.norm(np.abs(npEig_val-QREig_val), np.inf))
#     #loss of orthogonality

# print(tolerance_time_vector)
# print(error_eigenvalue)

# fig, axs=plt.subplots(2, 1, sharex=True, )
# axs[0].plot(eps, tolerance_time_vector)
# axs[0].set_ylabel('Time (s)')
# axs[0].set_xscale('log')
# axs[0].grid(True, which="both")

# axs[1].plot(eps, error_eigenvalue, marker='o')
# axs[1].set_ylabel('Eigenvalue error')
# axs[1].set_xscale('log')
# axs[1].set_yscale('log')
# axs[1].grid(True, which="both")
# axs[1].set_xlabel('Tolerance ($\epsilon$)', fontsize=15)
# plt.tight_layout()
# plt.show()