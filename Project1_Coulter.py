from __future__ import print_function, division   
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
import pandas as pd


def TriSolve(a,b,c,f):
    u = np.array([0.0,0.0,0.0,0.0])
    for i in range(1, len(b)):
        b[i] -= a[i]*c[i-1]/b[i-1]
        f[i] -= a[i]*f[i-1]/b[i-1]
    u[-1] = f[-1]/b[-1]
    for i in range(len(u)-1,0,-1):
        u[i-1] = (f[i-1] - c[i-1]*u[i])/b[i-1]
    return u

def main():
    dim=4
    matrix = np.zeros((dim,dim))
    a = np.array([0.0,2.0,2.0,2.0])
    b = np.array([3.0,3.0,3.0,3.0])
    c = np.array([1.0,1.0,1.0,0.0])
    f = np.array([1.0,2.0,3.0,4.0])
    for x in range(0, dim):
        matrix[x][x] = 3.0
    for x in range(1, dim):
        matrix[x][x-1] = 2.0  
    for x in range(0, dim-1):
        matrix[x][x+1] = 1.0
        
    eigval, eigvec = np.linalg.eig(matrix)
    print(eigval)
    print(eigvec)
        
    expected = np.linalg.solve(matrix, f)
    print("Expected: ", expected)
        
    calculated = TriSolve(a,b,c,f)
    print("Calculated: ", calculated)

    
main()