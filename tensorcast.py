import numpy as np
import tensorly as tl
import torch
import pandas as pd
tl.set_backend("pytorch")

def get_khatri_rao(X,factors,mode):
    X_1 = tl.unfold(X,mode)
    K = tl.tenalg.khatri_rao(factors, skip_matrix=mode)

    return tl.dot(X_1,K)   

def get_Hadamard(factors,r):
    H = tl.tensor(np.ones((r,r)))
    for fac in factors:
        H = H * tl.dot(tl.transpose(fac), fac)
    return H

def reconstruct_tensor(A,B,T):
    X = tl.cp_tensor.cp_to_tensor((None,[A,B,T]))
    Y = tl.cp_tensor.cp_to_tensor((None,[A,A,T]))
    return X,Y

def getError(X,Y,A,B,T,alpha):
    X_hat, Y_hat = reconstruct_tensor(A,B,T)
    error = tl.norm(X)**2 + tl.norm(X_hat)**2 -2*tl.tenalg.inner(X,X_hat) + alpha*(tl.norm(Y)**2+tl.norm(Y_hat)**2-2*tl.tenalg.inner(Y,Y_hat))
    return error

def getFit(X,Y,A,B,T,alpha):
    X_hat, Y_hat = reconstruct_tensor(A,B,T)
    fit = 1-(tl.norm(X-X_hat)/tl.norm(X)+alpha*tl.norm(Y-Y_hat)/tl.norm(Y))
    return fit

def CoupledTensorFac(A,B,T,N,M,K,r,d,alpha,maxiters,info):
    # Tensor Initialization
    X,Y = reconstruct_tensor(A,B,T)
    
    U1 = tl.tensor(np.random.rand(N,r))
    U2 = tl.tensor(np.random.rand(M,r))
    U3 = tl.tensor(np.random.rand(K,r))
    
    factorsX = [U1,U2,U3]
    factorsY = [U1,U1,U3]
    error = getError(X,Y,U1,U2,U3,alpha)
    print("Non-negative Couple Tensor Factorization")
    print(f'Initial Error : {error}')
    print("==========================================")
    
    # optimization
    for iter in range(maxiters):
#         print(U1[0])
        U1Num = get_khatri_rao(X,factorsX,0) + alpha*get_khatri_rao(Y,factorsY,1)
#         print(U1Num[0])
        U1Dem = tl.dot(U1,get_Hadamard([U2,U3],r)) + alpha*tl.dot(U1,get_Hadamard([U1,U3],r))
#         print(U1Dem[0])
        U1res = (U1Num/U1Dem)**(1/d)
#         print(U1res[0])
        U1 = U1*U1res
#         print(U1[0])

        factorsX[0] = U1
        factorsY[0] = U1
        factorsY[1] = U1

        U2Num = get_khatri_rao(X,factorsX,1)
        U2Dem = tl.dot(U2,get_Hadamard([U1,U3],r))
        U2res = (U2Num/U2Dem)**(1/d)
        U2 = U2*U2res

        factorsX[1] = U2

        U3Num = get_khatri_rao(X,factorsX,2) + alpha*get_khatri_rao(Y,factorsY,2)
        U3Dem = tl.dot(U3,get_Hadamard([U1,U2],r))+alpha*tl.dot(U3,get_Hadamard([U1,U1],r))
        U3res = (U3Num/U3Dem)**(1/d)
        U3 = U3*U3res

        factorsX[2] = U3
        factorsY[2] = U3
        
        error = getError(X,Y,U1,U2,U3,alpha)
        fit = getFit(X,Y,U1,U2,U3,alpha)
        if info:
            print(f'Iteration {iter+1} error : {error:.3f} \tRecon Tensor Fit : {fit:.3f}')
    print(f'Error : {error:.3f} \tRecon Tensor Fit : {fit:.3f}')    
        
        #print(T[iter])
    return U1,U2,U3

# Coupled Tensor Factorization
def RandomCoupledTensorFac(N,M,K,r,d,alpha,maxiters):
    # Tensor Initialization
    X = tl.tensor(np.random.rand(N,M,K))
    Y = tl.tensor(np.random.rand(N,N,K))
    
    # Initialization (random)
    A = tl.tensor(np.random.rand(N,r))
    B = tl.tensor(np.random.rand(M,r))
    T = tl.tensor(np.random.rand(K,r))
    factorsX = [A, B, T]
    factorsY = [A, A, T]
    error = getError(X,Y,A,B,T,alpha)
    print("Non-negative Couple Tensor Factorization")
    print(f'Initial Error : {error}')
    print("==========================================")
    
    # optimization
    for iter in range(maxiters):
        ANum = get_khatri_rao(X,factorsX,0) + alpha*get_khatri_rao(Y,factorsY,1)
        print(ANum[0])
        ADem = tl.dot(A,get_Hadamard([B,T],r)) + alpha*tl.dot(A,get_Hadamard([A,T],r))
        print(ADem[0])
        Ares = (ANum/ADem)**(1/d)
        print(Ares[0])
        A = A*Ares
        print(A[0])

        factorsX[0] = A
        factorsY[0] = A
        factorsY[1] = A

        BNum = get_khatri_rao(X,factorsX,1)
        BDem = tl.dot(B,get_Hadamard([A,T],r))
        Bres = (BNum/BDem)**(1/d)
        B = B*Bres

        factorsX[1] = B

        TNum = get_khatri_rao(X,factorsX,2) + alpha*get_khatri_rao(Y,factorsY,2)
        TDem = tl.dot(T,get_Hadamard([A,B],r))+alpha*tl.dot(T,get_Hadamard([A,A],r))
        Tres = (TNum/TDem)**(1/d)
        T = T*Tres

        factorsX[2] = T
        factorsY[2] = T
        
        oldError = error
        error = getError(X,Y,A,B,T,alpha)
        fit = getFit(X,Y,A,B,T,alpha)
        print(f'Iteration {iter+1} error : {error:.3f}\tError change : {(oldError-error):.3f} \tRecon Tensor Fit : {fit:.3f}')
    return A,B,T

# Export T
def saveFac(T,label):
    T = tl.to_numpy(T)
#     Tnp = pd.DataFrame(T)
    Tnp = np.save(f'/home/seyunkim/tensorcast_py/T{label}.npy',T)
