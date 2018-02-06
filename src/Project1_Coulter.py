from __future__ import print_function, division   
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.linalg as linalg
import pandas as pd
import time


def TriSolve(a,b,c,f,dim):
    u = np.zeros(dim)
    for i in range(1, dim):
        b[i] -= a[i]*c[i-1]/b[i-1]
        f[i] -= a[i]*f[i-1]/b[i-1]
    u[-1] = f[-1]/b[-1]
    for i in range(dim-1,0,-1):
        u[i-1] = (f[i-1] - c[i-1]*u[i])/b[i-1]
    return u

def main():
    dim = 5000
    h = 1/dim
    matrix = np.zeros((dim,dim))
    a = -np.ones(dim)
    a[0] = 0
    b = 2*np.ones(dim)
    c = -np.ones(dim)
    c[-1] = 0    
    f = np.ones(dim)
    j = 0
    while j <= dim-1:
        f[j] = 100*np.exp(-10*j*h)*h**2
        j += 1
    for x in range(0, dim):
        matrix[x][x] = 2.0
    for x in range(1, dim):
        matrix[x][x-1] = -1.0  
    for x in range(0, dim-1):
        matrix[x][x+1] = -1.0
        
#    eigval, eigvec = np.linalg.eig(matrix)
#    print(eigval)
    
    t0 = time.time()
    expected1 = np.linalg.solve(matrix, f)
#    print("Expected: ", expected1, "\n")
    t1 = time.time()
    total = t1-t0
    print("Numpy linalg time: ", total, "\n")
    
    t0 = time.time()
    LU = linalg.lu_factor(matrix)
    x = linalg.lu_solve(LU, f)
#    print("LU: ", x, "\n")
    t1 = time.time()
    total = t1-t0
    print("Scipy LU time: ", total, "\n")
    
    t0 = time.time()    
    calculated = TriSolve(a,b,c,f,dim)
#    print("Calculated: ", calculated)
    t1 = time.time()
    total = t1-t0
    print("Our code time: ", total, "\n")

#    print(np.isclose(expected1,calculated))

main()