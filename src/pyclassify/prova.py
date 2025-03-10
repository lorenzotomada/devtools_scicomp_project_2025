import numpy as np
from time import time
dim=10
T=5*np.diag(np.ones(dim))+20*np.diag(np.ones(dim-1), 1)+20*np.diag(np.ones(dim-1), -1)

t_s=time()
Q, R=np.linalg.eigh(T)
t_end=time()
print(np.linalg.eigh(T)[1])
print(np.linalg.eigh(T)[0])
print(f"Elapsed: {t_end-t_s}")
