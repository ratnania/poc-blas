#!/usr/bin/env python
# coding: utf-8

# In[3]:


from scipy.sparse.linalg import bicg
from scipy.linalg import blas
import numpy as np
from numpy import linalg as lg
from scipy.io import mmread
from time import time
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
# A1.shape=(1806, 1806) real symmetric positive definite
B3=np.array(mmread('/content/bcsstm20.mtx.gz').todense())
# A3.shape=(485,485) real symmetric positive definite
B4=np.array(mmread('/content/bcsstm24.mtx.gz').todense())
# A4.shape=(3562,3562) real symmetric positive definite
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


# #**BICG Method**

# In[4]:


def bicg_iter(A, b,x0,maxiter,tol):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1
    return bicg(A, b, callback=callback,tol = tol,maxiter=maxiter)[0],num_iters

def BICG(A,b,x0,tol,maxiter):
  C   = np.transpose(A)
  r0  = b - blas.dgemv(1.0, A, x0)
  rt0 = r0/( blas.dnrm2(r0)**2) 
  p0  = r0
  pt0 = rt0
  i   = 0
  while (i<maxiter):
    alpha = blas.ddot(r0,rt0)/ blas.ddot(blas.dgemv(1.0, A, p0), pt0)  
    x1    = x0 + (alpha*p0)
    r1    = r0 - blas.dgemv(alpha, A, p0)
    rt1   = rt0- blas.dgemv(alpha, C, pt0)   
    beta  = blas.ddot(r1,rt1)/ blas.ddot(r0,rt0)   
    p0    = r1 + beta*p0
    pt0   = rt1 + beta*pt0
    i     = i+1
    r0    = r1
    rt0   = rt1
    x0    = x1
    if blas.dnrm2(r1)<tol:
      break
  return x1,i


# # **Cluster 1:**

# In[5]:


matricies=[B5,C1,B9,C2]
tol=1e-5
itermax = 10000


# In[6]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = bicg_iter(matricies[i],b,x,itermax,tol)[1]  
  y=bicg(matricies[i],b,x,tol,itermax)[0]  
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= BICG(matricies[i], b, x, tol,itermax)   
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

# In[7]:


matricies=[C7,B10,A6,B11,C8,C9]
tol=1e-5
itermax = 10000


# In[8]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = bicg_iter(matricies[i],b,x,itermax,tol)[1]  
  y=bicg(matricies[i],b,x,tol,itermax)[0]  
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= BICG(matricies[i], b, x, tol,itermax)   
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

# In[11]:


matricies=[B6,B1,B4]
tol=1e-5
itermax = 100000


# In[12]:


my_data = []
for i in range(len(matricies)):
  l = []
  n = len(matricies[i])
  b=np.ones(n)
  x = np.zeros(n)
  sol  = lg.solve(matricies[i], b)
  itersc = bicg_iter(matricies[i],b,x,itermax,tol)[1]  
  y=bicg(matricies[i],b,x,tol,itermax)[0]  
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= BICG(matricies[i], b, x, tol,itermax)   
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


# # **Cluster 4:**

# In[ ]:


matricies=[C3,C4,C10,C12]
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
  itersc = bicg_iter(matricies[i],b,x,itermax,tol)[1]  
  y=bicg(matricies[i],b,x,tol,itermax)[0]  
  sc_err=np.linalg.norm(y-sol)
  sc_res=lg.norm(np.dot(matricies[i],y)-b)
  x,iterm= BICG(matricies[i], b, x, tol,itermax)   
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

