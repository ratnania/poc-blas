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
def transpose(A:'float[:,:]',C:'float[:,:]'):
  for i in range(A.shape[0]):
    for j in range(i,A.shape[1],1):
      C[i][j]=A[j][i]
      C[j][i]=A[i][j]
  return 0
################


def BICG(A1: 'float[:,:]',b: 'float[:]',x0:'float[:]',tol:'float',maxiter:'int'):
  n=len(b)
  C1= zeros(A1.shape)
  r0=zeros(n)
  p0=zeros(n)
  pt0=zeros(n)
  a= zeros(n)
  c= zeros(n)
  x1 = zeros(n)
  r1 = zeros(n)
  rt0 = zeros(n)
  rt1 = zeros(n)
  transpose(A1,C1)
  MatVec(A1,x0,r0)
  r0[:] = b[:] - r0[:]
  nr=norm2(r0)
  rt0[:]=r0[:]/nr
  p0[:]=r0[:]
  pt0[:]=rt0[:]
  i=0
  while (i<maxiter):
    MatVec(A1,p0,a)
    MatVec(C1,pt0,c)
    alpha=dot(r0,rt0)/dot(a,pt0)
    x1[:]=x0[:]+ alpha*p0[:]
    r1[:]=r0[:]-alpha*a[:]
    rt1[:]=rt0[:]-alpha*c[:]
    beta=dot(r1,rt1)/dot(r0,rt0)
    p0[:]=r1[:]+beta*p0[:]
    pt0[:]=rt1[:]+beta*pt0[:]
    i=i+1
    r0[:]=r1[:]
    rt0[:]=rt1[:]
    x0[:]=x1[:]
    nr = norm2(r1)
    if sqrt(nr)<tol:
      break
  return i


