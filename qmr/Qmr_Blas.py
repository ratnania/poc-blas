#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import numpy.linalg as lg
from scipy.sparse.linalg import qmr
from scipy.linalg import blas
from time import time
from tabulate import tabulate
from  scipy.io import mmread
A1=np.array(mmread('/content/west0067.mtx.gz').todense())
A2=np.array(mmread('/content/jpwh_991.mtx.gz').todense())
A5=np.array(mmread('/content/impcol_a.mtx.gz').todense())
A7=np.array(mmread('/content/impcol_e.mtx.gz').todense())
A8=np.array(mmread('/content/sherman4.mtx.gz').todense())
A10=np.array(mmread('/content/orsreg_1.mtx.gz').todense())
A11=np.array(mmread('/content/orsirr_1.mtx.gz').todense())
A12=np.array(mmread('/content/add20.mtx.gz').todense())
A13=np.array(mmread('/content/gre__185.mtx.gz').todense())
A14=np.array(mmread('/content/fs_680_3.mtx.gz').todense())
A15=np.array(mmread('/content/fs_680_2.mtx.gz').todense())
B1=np.array(mmread('/content/west0381.mtx.gz').todense())
B2=np.array(mmread('/content/e05r0200.mtx.gz').todense())


# # The Quasi Minimal Residual algorithm

# In[4]:


def qmr_iter(A, b,x0,maxiter,tol):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1
    return qmr(A, b, callback=callback,tol = tol,maxiter=maxiter),num_iters




def QMR(A, b, x,m, tol,Itermax):
    r      = b-blas.dgemv(1.0, A, x)
    n      = A.shape[0] 
    beta   = blas.dnrm2(r)
    V      = np.zeros((n,m+2))
    W      = np.zeros((n,m+2))
    V[:,1] = (r/beta) 
    W[:,1] = (r/beta)
    alphaH = np.zeros(m+1) 
    betaH  = np.zeros(m+2)                                                        
    deltaH = np.zeros(m+2)
    
    k  = 0
    while k < Itermax and beta>tol:   
        for j in range(1,m+1):
            alphaH[j]  =  blas.ddot(blas.dgemv(1.0, A, V[:,j]), W[:,j])                              
            vtelda     = blas.dgemv(1.0, A, V[:,j]) - alphaH[j]*V[:,j]- betaH[j]*V[:,j-1]  
            wtelda     =  blas.dgemv(1.0, A.T,W[:,j]) - alphaH[j]*W[:,j]- deltaH[j]*W[:,j-1]           
            deltaH[j+1]= abs(blas.ddot(vtelda,wtelda))**0.5      
            if deltaH[j+1]== 0:
                break
            else:
                betaH[j+1] = blas.ddot(vtelda,wtelda)/deltaH[j+1]
                W[:,j+1]   = wtelda/betaH[j+1]
                V[:,j+1]   = vtelda/deltaH[j+1]
        H      = np.diag(alphaH[1:m+1])+np.diag(betaH[2:m+1],1)+np.diag(deltaH[2:m+1],-1)
        em     = np.zeros(m)
        em[m-1]  =1
        Htelda = np.zeros((m+1,m))
        Htelda[:m,:]= H
        Htelda[m:m+1,:]  = deltaH[-1]*em
        #The QMR Algorithm for Linear Systems
        e1   = np.zeros(m+1)
        e1[0]= 1
        Vm   = V[:,1:m+1]
        Q, R = np.linalg.qr(Htelda,mode='complete') 
        Rm   = R[:m,:m] 
        g    = blas.dgemv(beta, Q.T,e1)
        gm   = g[:m]
        ym   = np.linalg.solve(Rm, gm)
        # l'itération  de la méthode
        x    = x+ blas.dgemv(1.0,Vm ,ym )      
        r    = b - blas.dgemv(1.0, A, x)  
        beta = blas.dnrm2(r)
        V[:, 1] = r/beta 
        W[:,1]  = r/beta 
        k     = k + 1  
    return x,k 
    


# # Test

# In[ ]:


n=100
D1 = 2*np.eye(n) +np.diag(np.ones(n-1),-1)-np.diag(np.ones(n-1),1)
print('codition umber :',lg.cond(D1))


# In[ ]:



b = np.ones(n)
x = np.ones(n)
m = 20
tol=1.e-5
Itermax  = n
xx,k11 = QMR(D1, b, x,m,tol, Itermax)
print('nombre d iterations',k11)
sol=np.linalg.solve(D1, b)
err=np.linalg.norm(xx-sol)
print("\n L'erreur de la méthode: \t ", err) 
y = qmr(D1,b)[0]
print('scipy error',np.linalg.norm(y-sol))


# In[ ]:


def qmr_iter(D1, b,x0,maxiter,tol):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1
    return qmr(D1, b, callback=callback,tol = tol,maxiter=maxiter),num_iters


# In[ ]:


yy = qmr_iter(D1,b,x,Itermax,tol)
print('scipy')
print('le nombre d iteration est',yy[1])
print('le residu',lg.norm(np.dot(D1,yy[0][0])-b))


# # **Cluster 1:**

# In[5]:


matricies=[A2,A8,A10,A12,A11,A15,A14]
tol=1e-5
itermax = 10000
m=20


# In[6]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = qmr_iter(matricies[i],b,x,itermax,tol)[1] 
  y=qmr(matricies[i],b,x,tol,itermax)[0]
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= QMR(matricies[i], b, x,m, tol,itermax)
  err=lg.norm(x-sol)
  res=lg.norm(np.dot(matricies[i],x)-b)
  l.append(matricies[i].shape)
  l.append(lg.cond(matricies[i]))
  l.append(tol)
  l.append(iterm)
  l.append(res)
  l.append(err)
  l.append(itersc)
  l.append(sc_res)
  l.append(sc_err)
  my_data.append(l)
# create header
head = ['matricies','conditio number','tol','Our_nbr_iter','res','Our_err','scipy_nbr_iter','sc_res','scipy_err']
  
# display table
print(tabulate(my_data, headers=head, tablefmt="grid")) 


# **Size of Krylov subspace: 30**

# In[8]:


matricies=[A2,A8,A10,A12,A11,A15,A14]
tol=1e-5
itermax = 10000
m=30


# In[9]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = qmr_iter(matricies[i],b,x,itermax,tol)[1] 
  y=qmr(matricies[i],b,x,tol,itermax)[0]
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= QMR(matricies[i], b, x,m, tol,itermax)
  err=lg.norm(x-sol)
  res=lg.norm(np.dot(matricies[i],x)-b)
  l.append(matricies[i].shape)
  l.append(lg.cond(matricies[i]))
  l.append(tol)
  l.append(iterm)
  l.append(res)
  l.append(err)
  l.append(itersc)
  l.append(sc_res)
  l.append(sc_err)
  my_data.append(l)
# create header
head = ['matricies','conditio number','tol','Our_nbr_iter','res','Our_err','scipy_nbr_iter','sc_res','scipy_err']
  
# display table
print(tabulate(my_data, headers=head, tablefmt="grid")) 


# # **Cluster 2:**

# In[10]:


matricies=[A1,A13,B2]
tol=1e-5
itermax = 1000
m=20


# In[11]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = qmr_iter(matricies[i],b,x,itermax,tol)[1] 
  y=qmr(matricies[i],b,x,tol,itermax)[0]
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= QMR(matricies[i], b, x,m, tol,itermax)
  err=lg.norm(x-sol)
  res=lg.norm(np.dot(matricies[i],x)-b)
  l.append(matricies[i].shape)
  l.append(lg.cond(matricies[i]))
  l.append(tol)
  l.append(iterm)
  l.append(res)
  l.append(err)
  l.append(itersc)
  l.append(sc_res)
  l.append(sc_err)
  my_data.append(l)
# create header
head = ['matricies','conditio number','tol','Our_nbr_iter','res','Our_err','scipy_nbr_iter','sc_res','scipy_err']
  
# display table
print(tabulate(my_data, headers=head, tablefmt="grid")) 


# # **Cluster 3:**

# In[13]:


matricies=[B1,A7,A5]
tol=1e-5
itermax = 1000
m=20


# In[14]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = qmr_iter(matricies[i],b,x,itermax,tol)[1] 
  y=qmr(matricies[i],b,x,tol,itermax)[0]
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= QMR(matricies[i], b, x,m, tol,itermax)
  err=lg.norm(x-sol)
  res=lg.norm(np.dot(matricies[i],x)-b)
  l.append(matricies[i].shape)
  l.append(lg.cond(matricies[i]))
  l.append(tol)
  l.append(iterm)
  l.append(res)
  l.append(err)
  l.append(itersc)
  l.append(sc_res)
  l.append(sc_err)
  my_data.append(l)
# create header
head = ['matricies','conditio number','tol','Our_nbr_iter','res','Our_err','scipy_nbr_iter','sc_res','scipy_err']
  
# display table
print(tabulate(my_data, headers=head, tablefmt="grid")) 

