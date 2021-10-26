from numpy import zeros
from numpy import sqrt
###############
def norm2(v:'float[:]'):
    norme=0.0
    for i in range(v.shape[0]) :
        norme+=v[i]*v[i]
    return norme
################
def MatVec(A:'float[:,:]',b:'float[:]',c:'float[:]'):
  row,col=A.shape
  for i in range(row):
    s = 0.0
    for j in range(col):
      s+=A[i][j]*b[j]
    c[i] = s
  return 0
#################
def dot(a:'float[:]',b:'float[:]'):
  s=0.0
  for i in range(len(a)):
    s+=a[i]*b[i]
  return s
###############

def BICGSTAB(A:'float[:,:]',b:'float[:]',x0:'float[:]',tol:'float',maxiter:'int'):
  
  n=len(b)
  r0=zeros(n)
  rt=zeros(n)
  p0=zeros(n)
  s0=zeros(n)
  r1=zeros(n)
  a1=zeros(n)
  a2=zeros(n)
  a3=zeros(n)
  MatVec(A,x0,a1)
  r0[:]=b[:]-a1[:]
  rt[:]=r0[:]/norm2(r0)
  p0[:]=r0[:]
  for i in range(maxiter):
    MatVec(A,p0,a2)
    alpha=dot(r0,rt)/dot(a2,rt)
    s0[:]=r0[:]-alpha*a2[:]
    MatVec(A,s0,a3)
    w0=dot(a3,s0)/norm2(a3)
    x0[:]=x0[:]+alpha*p0[:]+w0*s0[:]
    r1[:]=s0[:]-w0*a3[:]
    beta=(dot(r1,rt)/dot(r0,rt))*(alpha/w0)
    p0[:]=r1[:]+beta*(p0[:]-w0*a2[:])
    r0[:]=r1[:]
    if sqrt(norm2(r1))<tol:
      break
  return i


