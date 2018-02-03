from __future__ import print_function, division   
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
import pandas as pd
import time


def TriSolve(a,b,c,f,dim):
    u = np.zeros(dim)
    for i in range(1, len(b)):
        b[i] -= a[i]*c[i-1]/b[i-1]
        f[i] -= a[i]*f[i-1]/b[i-1]
    u[-1] = f[-1]/b[-1]
    for i in range(len(u)-1,0,-1):
        u[i-1] = (f[i-1] - c[i-1]*u[i])/b[i-1]
    return u

def main():
    dim = 10
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
        
    #eigval, eigvec = np.linalg.eig(matrix)
    #print(eigval)
        
    expected = np.linalg.solve(matrix, f)
    print("Expected: ", expected)
        
    calculated = TriSolve(a,b,c,f,dim)
    print("Calculated: ", calculated)

    print(np.isclose(expected,calculated))

main()