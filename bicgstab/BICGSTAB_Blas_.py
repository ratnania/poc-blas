#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.sparse.linalg import bicgstab
from scipy.linalg import blas
import numpy as np
from numpy import linalg as lg
from time import time
from scipy.io import mmread
from tabulate import tabulate
from scipy.sparse import coo_matrix


A1=np.array(mmread('/content/bcsstk01.mtx.gz').todense())
#A1.shape=(48,46) Non symmetrique
A2=np.array(mmread('/content/impcol_c.mtx.gz').todense())
#A2.shape=(137,137) Non symmetrique
A3=np.array(mmread('/content/impcol_a.mtx.gz').todense())
#A3.shape=(207,207) Non symmetrique
A4=np.array(mmread('/content/mcca.mtx.gz').todense())
#A4.shape=(180, 180) Non symmetrique
A5=np.array(mmread('/content/bcsstm24.mtx.gz').todense())
#A5.shape=(3562,3562) Symmetrique definie positive
A6=np.array(mmread('/content/bcsstm19.mtx.gz').todense())
#A6.shape=(817,817) Symmetrique definie positive

B1=np.array(mmread('/content/bcsstk14.mtx.gz').todense())             
# B1.shape=(1806, 1806) real symmetric positive definite
B3=np.array(mmread('/content/bcsstm20.mtx.gz').todense())
# B3.shape=(485,485) real symmetric positive definite
B5=np.array(mmread('/content/bcsstm21.mtx.gz').todense())
B6=np.array(mmread('/content/bcsstk16.mtx.gz').todense())
B7=np.array(mmread('/content/bcsstk17.mtx.gz').todense())
B8=np.array(mmread('/content/bcsstk18.mtx.gz').todense())
B9=np.array(mmread('/content/bcsstm22.mtx.gz').todense())
B10=np.array(mmread('/content/bcsstk27.mtx.gz').todense())
B11=np.array(mmread('/content/bcsstm26.mtx.gz').todense())

C1=np.array(mmread('/content/west0067.mtx.gz').todense())              
C2=np.array(mmread('/content/jpwh_991.mtx.gz').todense())
C3=np.array(mmread('/content/impcol_a.mtx.gz').todense())
C4=np.array(mmread('/content/impcol_e.mtx.gz').todense())
C5=np.array(mmread('/content/sherman4.mtx.gz').todense())
C6=np.array(mmread('/content/orsreg_1.mtx.gz').todense())
C7=np.array(mmread('/content/orsirr_1.mtx.gz').todense())
C8=np.array(mmread('/content/add20.mtx.gz').todense())
C9=np.array(mmread('/content/gre__185.mtx.gz').todense())
C10=np.array(mmread('/content/fs_680_3.mtx.gz').todense())
C11=np.array(mmread('/content/fs_680_2.mtx.gz').todense())
C12=np.array(mmread('/content/west0381.mtx.gz').todense())
C13=np.array(mmread('/content/e05r0200.mtx.gz').todense())


# #**BICGSTAB  Method**
# 
# 
# 
# 

# In[ ]:



def BICGSTAB(A,b,x0,tol,maxiter):
  r0 = b- blas.dgemv(1.0, A, x0)
  rt = r0/( blas.dnrm2(r0)**2)    
  
  p0 = r0
  for i in range(maxiter):
    alpha = blas.ddot(r0,rt)/ blas.ddot(blas.dgemv(1.0, A, p0), rt)       
    s0    = r0-blas.dgemv(alpha, A, p0)    
    w0    = blas.ddot(blas.dgemv(1.0, A, s0), s0) /blas.ddot(blas.dgemv(1.0, A, s0), blas.dgemv(1.0, A, s0) )              
    x0    = x0 + alpha*p0 + blas.ddot(w0,s0)    
    r1    = s0 - blas.ddot(w0, blas.dgemv(1.0, A, s0))  
    beta  = (alpha*blas.ddot(r1,rt)/w0*blas.ddot(r0,rt))   
    p0    = r1 + beta*(p0-blas.ddot(w0, blas.dgemv(1.0, A, p0)))      
    r0    = r1
    if blas.dnrm2(r1)<tol:    
      break
  return x0,i+1

def bicgstab_iter(A, b,x0,tol,maxiter):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1
    return bicgstab(A, b, callback=callback,tol = tol,maxiter=maxiter,x0=x0)[0],num_iters


# # **Cluster 1:**

# In[ ]:


matricies=[B5,C1,B9,C2]
tol=1e-5
itermax = 10000


# In[ ]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = bicgstab_iter(matricies[i],b,x,tol,itermax)[1] 
  y=bicgstab(matricies[i],b,x,tol,itermax)[0] 
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= BICGSTAB(matricies[i], b, x, tol,itermax)  
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

# In[ ]:


matricies=[C7,B10,A6,B11,C8,C9]
tol=1e-5
itermax = 10000


# In[ ]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = bicgstab_iter(matricies[i],b,x,tol,itermax)[1] 
  y=bicgstab(matricies[i],b,x,tol,itermax)[0] 
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= BICGSTAB(matricies[i], b, x, tol,itermax)  
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

# In[ ]:


matricies=[B5,C1,B9,A6,B3,B6,A1,B10]
tol=1e-5
itermax = 10000


# In[ ]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = bicgstab_iter(matricies[i],b,x,tol,itermax)[1] 
  y=bicgstab(matricies[i],b,x,tol,itermax)[0] 
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= BICGSTAB(matricies[i], b, x, tol,itermax)  
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


# In[ ]:


matricies=[B1,B11]
tol=1e-5
itermax = 100000


# In[ ]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = bicgstab_iter(matricies[i],b,x,tol,itermax)[1] 
  y=bicgstab(matricies[i],b,x,tol,itermax)[0] 
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= BICGSTAB(matricies[i], b, x, tol,itermax)  
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

