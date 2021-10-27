# BLAS

## [Matrix Storage](https://www.netlib.org/lapack/lug/node121.html)

BLAS/LAPACK allows the following different storage schemes for matrices:

* conventional storage in a two-dimensional array;
* packed storage for symmetric, Hermitian or triangular matrices;
* band storage for band matrices;

### [Conventional Storage](https://www.netlib.org/lapack/lug/node122.html)

The default scheme for storing matrices is the obvious one described [here](https://www.netlib.org/lapack/lug/node116.html#subsecarrayargs): a matrix A is stored in a two-dimensional array A, with matrix element aij stored in array element A(i,j).

If a matrix is triangular (upper or lower, as specified by the argument UPLO), only the elements of the relevant triangle are accessed. The remaining elements of the array need not be set. Such elements are indicated by $\ast$ in the examples below.

Similarly, if the matrix is upper Hessenberg, elements below the first subdiagonal need not be set.

Routines that handle symmetric or Hermitian matrices allow for either the upper or lower triangle of the matrix (as specified by UPLO) to be stored in the corresponding elements of the array; the remaining elements of the array need not be set. 

### [Packed Storage](https://www.netlib.org/lapack/lug/node123.html) 

Symmetric, Hermitian or triangular matrices may be stored more compactly, if the relevant triangle (again as specified by UPLO) is packed by columns in a one-dimensional array. In LAPACK, arrays that hold matrices in packed storage, have names ending in `P'. 

### [Band Storage](https://www.netlib.org/lapack/lug/node124.html)

An m-by-n band matrix with kl subdiagonals and ku superdiagonals may be stored compactly in a two-dimensional array with kl+ku+1 rows and n columns. Columns of the matrix are stored in corresponding columns of the array, and diagonals of the matrix are stored in rows of the array. This storage scheme should be used in practice only if $kl, ku \ll \min(m,n)$, although LAPACK routines work correctly for all values of kl and ku. In LAPACK, arrays that hold matrices in band storage have names ending in `B'. 

## [Quick Reference Guide to the BLAS](https://www.netlib.org/lapack/lug/node145.html)

### Level 1

| subroutine  | operation                         | precisions  |
| ----------- | --------------------------------- | ----------- |
| ?ROTG       | Generate plane rotation           | S,D         |
| ?ROTMG      | Generate modified plane rotation  | S,D         |
| ?ROT        | Apply plane rotation              | S,D         |
| ?ROTM       | Apply modified plane rotation     | S,D         |
| ?SWAP       |                                   | S,D,C,Z     |
| ?SCAL       |                                   | S,D,C,Z     |
| ?COPY       |                                   | S,D,C,Z     |
| ?AXPY       |                                   | S,D,C,Z     |
| ?DOT        |                                   | S,D-DS      |
| ?DOTU       |                                   | C,Z         |
| ?DOTC       |                                   | C,Z         |
| ??DOT       |                                   | SDS         |
| ?NRM2       |                                   | S,D,SC,DZ   |
| ?ASUM       |                                   | S,D,SC,DZ   |
| I?AMAX      |                                   | S,D,C,Z     |

### Level 2 

| subroutine  | operation   | precisions   |
| ----------- | ----------- | -----------  |
| ?GEMV       |             | S,D,C,Z      |
| ?GBMV       |             | S,D,C,Z      |
| ?HEMV       |             | C,Z          |
| ?HBMV       |             | C,Z          |
| ?HPMV       |             | C,Z          |
| ?SYMV       |             | S,D          |
| ?SBMV       |             | S,D          |
| ?SPMV       |             | S,D          |
| ?TRMV       |             | S,D,C,Z      |
| ?TBMV       |             | S,D,C,Z      |
| ?TPMV       |             | S,D,C,Z      |
| ?TRSV       |             | S,D,C,Z      |
| ?TBSV       |             | S,D,C,Z      |
| ?TPSV       |             | S,D,C,Z      |
| ?GER        |             | S,D          |
| ?GERU       |             | C,Z          |
| ?GERC       |             | C,Z          |
| ?HER        |             | C,Z          |
| ?HPR        |             | C,Z          |
| ?HER2       |             | C,Z          |
| ?HPR2       |             | C,Z          |
| ?SYR        |             | S,D          |
| ?SPR        |             | S,D          |
| ?SYR2       |             | S,D          |
| ?SPR2       |             | S,D          |

### Level 3 

| subroutine  | operation   | precisions   |
| ----------- | ----------- | -----------  |
| ?GEMM       |             | S,D,C,Z      |
| ?SYMM       |             | S,D,C,Z      |
| ?HEMM       |             | C,Z          |
| ?SYRK       |             | S,D,C,Z      |
| ?HERK       |             | C,Z          |
| ?SYR2K      |             | S,D,C,Z      |
| ?HER2K      |             | C,Z          |
| ?TRMM       |             | S,D,C,Z      |
| ?TRSM       |             | S,D,C,Z      |

### Notes
