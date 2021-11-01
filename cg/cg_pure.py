from numpy import zeros

# ======================================================================
def norm2(v:'float[:]'):
    norme=0.0
    for i in range(v.shape[0]) :
        norme+=v[i]*v[i]
    return norme

# ======================================================================
def MatVec(A:'float[:,:]',b:'float[:]',c:'float[:]'):
  row,col=A.shape
  for i in range(row):
    s = 0.0
    for j in range(col):
      s+=A[i][j]*b[j]
    c[i] = s
  return 0

# ======================================================================
def dot(a:'float[:]',b:'float[:]'):
  s=0.0
  for i in range(len(a)):
    s+=a[i]*b[i]
  return s

# ======================================================================
def cg(A0:'float[:,:]',b:'float[:]',x0:'float[:]',maxiter:'int',tol:'float'):
    n=len(b)
    r=zeros(n)
    a1=zeros(n)
    a2=zeros(n)
    p=zeros(n)
    rpr= zeros(n)
    MatVec(A0,x0,a1)
    r[:]=b[:]-a1[:]
    p[:]=r[:]
    k=0
    while norm2(r) > tol and k<maxiter:
        MatVec(A0,p,a2)
        alpha = norm2(r)/dot(a2,p)
        x0[:]  = x0[:] + alpha*p[:]
        rpr[:]   = r[:]
        r[:]  = rpr[:] - alpha*a2[:]
        beta  = norm2(r)/norm2(rpr)
        p[:]     = r[:] + beta*p[:]
        k     = k + 1
    return k
