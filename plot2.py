import matplotlib.pyplot as plt
import numpy as np
import plot
import time
import scipy.linalg

cpp = True
if cpp:
    data = np.array([
        [25e3, 1e-3],
        [50e3, 2e-3],
        [100e3, 5e-3],
        [250e3, 10e-3],
        [500e3, 20e-3],
        [1e6, 43e-3],
        [2.5e6, 114e-3],
        [5e6, 218e-3],
        [10e6, 454e-3],
        [25e6, 1.11e0],
        [50e6, 2.29e0],
        [100e6, 5.38e0],
        [250e6, 27.63e0],
    ])

    plt.loglog(data[:,0],data[:,1],label="C++ Tridiag Sym")


np_sizes = np.logspace(1,3.5,num=20,dtype='int')
np_times = []
for dim in np_sizes:
    print("dim: ",dim)
    
    f = np.ones(dim)
    for j in range(dim):
        f[j] = 100*np.exp(-10*j/dim)/dim**2

    matrix = 2*np.diag(np.ones(dim)) - np.diag(np.ones(dim-1), 1) \
            - np.diag(np.ones(dim-1), -1)
    
    start = time.perf_counter()
    for i in range(10):
        print("numpy i: ",i)
        np.linalg.solve(matrix,f)
    end = time.perf_counter()
    np_times.append((end-start)/10)

#------------------#
# LU decomp method #
LU_times = []
LU_sizes = np.logspace(1,4.2,num=20,dtype='int')
for dim in LU_sizes:
    print("dim: ",dim)
    
    f = np.ones(dim)
    for j in range(dim):
        f[j] = 100*np.exp(-10*j/dim)/dim**2

    matrix = 2*np.diag(np.ones(dim)) - np.diag(np.ones(dim-1), 1) \
            - np.diag(np.ones(dim-1), -1)
    
    LU = scipy.linalg.lu_factor(matrix)
    start = time.perf_counter()
    for i in range(10):
        print("LU i: ",i)
        x = scipy.linalg.lu_solve(LU, f)
    end = time.perf_counter()
    LU_times.append((end-start)/10)

#------------------#
# Tridiag special method #
tdsp_times = []
tdsp_sizes = np.logspace(1,6,num=20,dtype='int')
for dim in tdsp_sizes:
    print("dim: ",dim)
    
    f = np.ones(dim)
    for j in range(dim):
        f[j] = 100*np.exp(-10*j/dim)/dim**2
    
    start = time.perf_counter()
    for i in range(10):
        print("tridiag_sp i: ",i)
        plot.tridiag_special(f)
    end = time.perf_counter()
    tdsp_times.append((end-start)/10)

'''    
#------------------#
# Tridiag method #
td_times = []
td_sizes = np.logspace(1,3.5,num=20,dtype='int')
for dim in tdsp_sizes:
    print("dim: ",dim)
    
    a = -np.ones(dim-1)
    b = 2*np.ones(dim)
    c = -np.ones(dim-1)
    
    f = np.ones(dim)
    for j in range(dim):
        f[j] = 100*np.exp(-10*j/dim)/dim**2
    
    start = time.perf_counter()
    for i in range(10):
        print("tridiag i: ",i)
        plot.tridiag(a,b,c,f)
    end = time.perf_counter()
    td_times.append((end-start)/10)
'''
td_sizes = [10, 50, 100, 500, 1000, 5000, 10000,
            50000, 100000, 500000, 1000000]

td_times = [5.1021575927734375e-05,
0.0001785755157470703,
0.00038313865661621094,
0.0010585784912109375,
0.003953695297241211,
0.010825634002685547,
0.04216814041137695,
0.10490608215332031,
0.2069101333618164,
1.0504841804504395,
2.1423940658569336]

plt.loglog(np_sizes,np_times,label="NumPy General")
plt.loglog(LU_sizes,LU_times,label="SciPy LU")
plt.loglog(tdsp_sizes,tdsp_times,label="Tridiag Special")
plt.loglog(td_sizes,td_times,label="Tridiag")
plt.xlabel("Matrix Size")
plt.ylabel("Time (s)")
plt.grid(which='minor', linestyle='--', linewidth=0.5)
plt.grid(which='major', linestyle='-', linewidth=0.75)
plt.legend()
plt.tight_layout()
plt.savefig("timing.pdf")
plt.show()