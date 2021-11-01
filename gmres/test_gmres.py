#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.sparse.linalg import cg, gmres, bicgstab, minres
import numpy.linalg as lg
from scipy.sparse.linalg import cg as cg_scipy
from scipy.linalg import blas
from time import time
from tabulate import tabulate
from scipy.io import mmread
#from gmres_blas import gmres as gmres_pyccel
#from gmres_pure import gmres as gmres_pure

# ======================================================================
def gmres_old(A, b, x, m, itermax, tol):
    r      = b-blas.dgemv(1.0, A, x)
    beta   = blas.dnrm2(r)
    ris    = np.zeros(itermax)
    V      = np.zeros((len(b), m+1))
    V[:,0] = r/beta
    H      = np.zeros((m+1,m))
    k      = 0
    while  k < itermax and beta>tol:
        ris[k]   = beta
        for j in range(m):
            w = blas.dgemv(1.0, A, V[:, j])
            for i in range(j+1):
                H[i,j]= blas.ddot(w,V[:,i])
                w     = w - H[i,j]*V[ :,i]
            H[j+1,j] = blas.dnrm2(w)
            if H[j+1,j]==0:
                break
            V[:, j+1] = w / H[j+1,j]
        e1      = np.zeros(m+1)
        e1[0]   = 1
        Vm      = V[:,:m]                       # Vm obtenue en supprimant la dernière colonne de V
        Q, R    = lg.qr(H,mode='complete')      # application du QR à la matrice H de taille (m+1)*m
        Rm      = R[:m,:m]
        C       =   np.transpose(Q)             # matrice extraite en supprimant la dernière colonne de R
        g       = blas.dgemv(beta, C, e1)       # vecteur de taille m+1
        gm      = g[:m]                         # vecteur de taille m (en garde les m premières composantes)
        ym      = lg.solve(Rm, gm)
        x       = x + blas.dgemv(1.0,Vm ,ym )
        r       = b - blas.dgemv(1.0, A, x)
        beta    = blas.dnrm2(r)
        V[:, 0] = r/beta
        k       = k + 1
    return x, k

# ======================================================================
def solve_sparse_scipy(A, b, m=30, tol=1.e-5, maxiter=3000, M=None, atol=None,
                       solver='cg', x0=None, xref=None):

    # ...
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters+=1
    # ...

    t1 = time()

    if solver == 'cg':
        x, status = cg(A, b, tol=tol, maxiter=maxiter,
                       M=M, callback=callback, atol=atol, x0=x0)

    elif solver == 'bicgstab':
        x, status = bicgstab(A, b, tol=tol, maxiter=maxiter,
                             M=M, callback=callback, atol=atol, x0=x0)

    elif solver == 'minres':
        x, status = minres(A, b, tol=tol, maxiter=maxiter,
                           M=M, callback=callback, x0=x0, shift=-3.)

    elif solver == 'gmres':
        x, status = gmres(A, b, tol=tol, maxiter=maxiter,
                          M=M, callback=callback, atol=atol, restart=m, x0=x0)

    else:
        raise NotImplemented('solver not available')

    t2 = time()
    res_norm = np.linalg.norm(A.dot(x) - b)
    err_norm = None
    if not( xref is None ):
        err_norm   = np.linalg.norm(x - xref)
    info = {'niter': num_iters, 'res_norm': res_norm, 'err_norm': err_norm, 'time': (t2-t1)*1000}

    return x, info

# ======================================================================
def solve_sparse_pyccel(A, b, m=30, tol=1.e-5, maxiter=3000, M=None, atol=None,
                        solver='gmres', x0=None, xref=None):

    # TODO
    if not( M is None):
        raise ValueError('M must be None')

    # TODO
    if not( atol is None):
        raise ValueError('atol must be None')

    # ...
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters+=1
    # ...

    x = x0.copy()

    t1 = time()
    if solver == 'gmres':
        err_r, num_iters = gmres_pyccel(A, b, x, maxiter=maxiter, tol=tol, m=m)

    elif solver == 'gmres_pure':
        num_iters = gmres_pure(A, b, x, m, maxiter, tol)

    elif solver == 'gmres_old':
        x, num_iters = gmres_old(A, b, x, m, maxiter, tol)

    else:
        raise NotImplemented('solver not available')

    t2 = time()
    res_norm = np.linalg.norm(A.dot(x) - b)
    err_norm = None
    if not( xref is None ):
        err_norm   = np.linalg.norm(x - xref)
    info = {'niter': num_iters, 'res_norm': res_norm, 'err_norm': err_norm, 'time': (t2-t1)*1000}

    return x, info

# ======================================================================
def run(A, tol=1.e-5, maxiter=100):
    table = []
    np.random.seed(2021)

    n  = A.shape[0]
    x  = np.random.random(n)
    b  = A @ x
    x0 = np.zeros(n)

    x_scipy, info = solve_sparse_scipy(A, b, tol=tol, maxiter=maxiter, x0=x0, xref=x)
    line = ['scipy', info['niter'], info['res_norm'], info['err_norm'], info['time']] ; table.append(line)

#    x_pyccel, info = solve_sparse_pyccel(A, b, tol=tol, maxiter=maxiter, x0=x0, xref=x)
#    line = ['pyccel-blas', info['niter'], info['res_norm'], info['err_norm'], info['time']] ; table.append(line)

#    x_pyccel, info = solve_sparse_pyccel(A, b, tol=tol, maxiter=maxiter, x0=x0, xref=x, solver='gmres_pure')
    x_pyccel, info = solve_sparse_pyccel(A, b, tol=tol, maxiter=maxiter, x0=x0, xref=x, solver='gmres_old')
    line = ['pyccel-pure', info['niter'], info['res_norm'], info['err_norm'], info['time']] ; table.append(line)

    return table

# create header
headers = ['implementation', 'num_iters','res','err','time(ms)']

from scipy.sparse import diags
#for n in [200, 500, 1000, 2000, 3000, 4000, 6000]:
for n in [200]:
    print('>>> n = ', n)
    A = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).toarray()

    table = run(A)

    # display table
    print(tabulate(table, headers=headers, tablefmt="grid"))

