#Tridiag solver

def tridiag_solve(M,b):
        #Eliminate lower diagonal & normalize diagonal
        for i in range(M.shape[0]-1):
            #Normalize diagonal element
            norm = M[i,i]+0. # +0. since python is dumb
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