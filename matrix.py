#Tridiag solver

import numpy as np

def banded(Aa,va,up,down):
    A = np.copy(Aa)
    v = np.copy(va)
    N = len(v)
    
    # Gaussian elimination
    for m in range(N):
        # Normalization factor
        div = A[up,m]
        
        # Update vector
        v[m] /= div
        for k in range(1,down+1):
            if m+k<N:
                v[m+k] -= A[up+k,m]*v[m]
        
        # Normalize and subtract pivot row
        for i in range(up):
            j = m+up-i
            if j<N:
                A[i,j] /= div
                for k in range(1,down+1):
                    A[i+k,j] -= A[up+k,m]*A[i,j]
        
        # Backsubstitution
        for m in range(N-2,-1,-1):
            for i in range(up):
                j = m+up-i
                if j<N:
                    v[m] -= A[i,j]*v[j]
        
        return v

def tridiag_solve(M,f):
    size = M.shape[0]
    
    #Forward Propagation
    for i in range(1,size):
        scale = M[i][i-1]/M[i-1][i-1]
        M[i][i] -= scale*M[i-1][i]
        f[i] -= scale*f[i-1]
    
    sol_vector = np.zeros_like(f)
    sol_vector[size-1] = f[size-1]/M[size-1][size-1]
    
    #Backward Propagation
    for i in range(size-2,-1,-1):
        sol_vector[i] = (f[i]-M[i][i+1]*sol_vector[i+1])/M[i][i]
    
    return sol_vector

'''
M = np.array([
    [1234,1252,0,0],
    [897546,13454,63542,0],
    [0,8764534,586743,54736],
    [0,0,68975643,9875643]],dtype=float)

f = np.array([1,2,3,4],dtype=float)
'''
M = np.array([[2,-1,0,0], [-1,2,-1,0], [0,-1,2,-1], [0,0,-1,2]])
f = np.array([123,312,654,423])

print(np.linalg.solve(M,f), "\nand\n", tridiag_solve(M,f), "\nand\n", banded(M,f,1,1))

'''
def tridiag_solve(M,b):
        
        #Eliminate lower diagonal & normalize diagonal
        for i in range(M.shape[0]-1):
            
            #Normalize diagonal element
            norm = M[i,i]
            M[i] /= norm
            b[i] /= norm
            
            #Eliminate lower diagonal element
            scale = M[i+1,i]
            M[i+1] -= scale*M[i]
            b[i+1] -= scale*b[i]
            
        #Normalize last row
        norm = M[-1,-1]
        M[-1] /= norm
        b[-1] /= norm
        
        
        #Backpropagate (more efficient than row operations)
        x = np.zeros(len(b))
        for i in range(len(x)-1,-1,-1):
            x[i] = b[i]
            for j in range(i+1,len(x)):
                x[i] -= M[i,j]*x[j]
            
        return x
'''