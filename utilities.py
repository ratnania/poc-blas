import numpy as np

# ==============================================================================
def symmetrize(a, lower=False):
    n = a.shape[0]
    if lower:
        for j in range(n):
            for i in range(j):
                a[i,j] = a[j,i]
    else:
        for i in range(n):
            for j in range(i):
                a[i,j] = a[j,i]

    return a

# ==============================================================================
def triangulize(a, lower=False):
    n = a.shape[0]
    if lower:
        for j in range(n):
            for i in range(j):
                a[i,j] = 0.
    else:
        for i in range(n):
            for j in range(i):
                a[i,j] = 0.

    return a

# ==============================================================================
def general_to_band(kl, ku, a):
    n = a.shape[1]
    ab = np.zeros((kl+ku+1, n), dtype=a.dtype)

    for j in range(n):
        k = ku - j
        i1 = max (j - ku, 0)
        i2 = min (j + kl + 1, n)
        for i in range(i1, i2):
            ab[k+i,j] = a[i,j]

    return ab

# ==============================================================================
def general_to_packed(a, lower=False):
    n = a.shape[0]
    ap = np.zeros(n*(n+1)//2, dtype=a.dtype)
    if lower:
        k = 0
        for j in range(n):
            for i in range(j,n):
                ap[k] = a[i,j]
                k += 1
    else:
        k = 0
        for j in range(n):
            for i in range(j+1):
                ap[k] = a[i,j]
                k += 1

    return ap
