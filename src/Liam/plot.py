import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import time

if False:
    data1 = np.transpose(np.loadtxt("initial.tsv"))
    data2 = np.transpose(np.loadtxt("solution.tsv"))
    data3 = np.transpose(np.loadtxt("analyt.tsv"))

    positions = np.linspace(0,1,data1.shape[0])

    plt.plot(data1)
    plt.show()
    plt.plot(data2)
    plt.plot(data3)
    plt.show()

    error = np.log10(np.fabs((data2-data3)/data3))
    plt.plot(positions,error)
    plt.ylabel("Logarithmic Error")
    plt.xlabel("Position")
    plt.tight_layout()
    plt.savefig("error.pdf")
    plt.show()


#--------------------------------------#
#          Python Methods              #
#--------------------------------------#

dim = 2000
h = 1/dim

a = -np.ones(dim-1)
b = 2*np.ones(dim)
c = -np.ones(dim-1)
f = np.ones(dim)

for j in range(dim):
    f[j] = 100*np.exp(-10*j*h)*h**2

matrix = 2*np.diag(np.ones(dim)) - np.diag(np.ones(dim-1), 1) \
        - np.diag(np.ones(dim-1), -1)

import scipy.linalg
def LU_solve(matrix, f):
    f = copy.deepcopy(f)
    
    LU = scipy.linalg.lu_factor(matrix)
    x = scipy.linalg.lu_solve(LU, f)
    
    return x

def gaussian(matrix, f):
    f = copy.deepcopy(f)
    return np.linalg.solve(matrix,f)
    
def tridiag_special(f):
    dim = f.size
    
    diag = np.ones(dim)
    sol = np.ones(dim)
    
    f = copy.deepcopy(f)
    
    for i in range(0,dim):
        diag[i] = (i+2.)/(i+1.)
    
    for i in range(1,dim):
        f[i] += (f[i-1]/diag[i-1])
        
    sol[dim-1] = f[dim-1]/diag[dim-1]
    
    for j in range(dim-2,-1,-1):
        sol[j] = (f[j]+sol[j+1])/diag[j]
    
    return sol

def tridiag(a,b,c,f):
    dim = f.size
    u = np.zeros(dim)
    '''
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    c = copy.deepcopy(c)
    f = copy.deepcopy(f)
    '''
    # Forward substitution
    for i in range(0, dim-1):
        temp = a[i]/b[i]
        b[i+1] -= temp*c[i]
        f[i+1] -= temp*f[i]
    
    # Backward substitution
    u[-1] = f[-1]/b[-1]
    for i in range(dim-1,0,-1):
        u[i-1] = (f[i-1] - c[i-1]*u[i])/b[i-1]
    
    return u

if __name__ == "__main__":
    # Analytic Solution
    positions = np.linspace(0,1,dim)
    analytic = 1-(1-np.exp(-10))*positions - np.exp(-10*positions)

    #LU_soln = LU_solve(matrix, f)
    gauss_soln = gaussian(matrix, f)
    tridiag_sp_soln = tridiag_special(f)
    tridiag_soln = tridiag(a,b,c,f)


    #LU_error = np.log10(np.fabs((LU_soln-analytic)/analytic))
    gauss_error = np.log10(np.fabs((gauss_soln-analytic)/analytic))
    tridiag_sp_error = np.log10(np.fabs((tridiag_sp_soln-analytic)/analytic))
    tridiag_error = np.log10(np.fabs((tridiag_soln-analytic)/analytic))

    #plt.plot(positions, LU_error, 'b.', markersize = 0.5)
    #plt.plot(positions, gauss_error, '.', markersize = 1, label="gauss")
    #plt.plot(positions, tridiag_sp_error, 'r.', markersize = 1, label="LU, gauss, tridiag_sp")
    plt.plot(positions, tridiag_error, 'm.', markersize = 1, label="Python")

    data2 = np.transpose(np.loadtxt("solution.tsv"))
    error = np.log10(np.fabs((data2-analytic)/analytic))
    plt.plot(positions, error, 'g.', markersize = 1, label="C++")

    plt.title("N = "+str(dim)+" Errors")
    plt.xlabel("Position")
    plt.ylabel("Logarithmic Error")
    plt.ylim(-3,0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("error.pdf")
    plt.show()