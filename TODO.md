# BUGS

* this import raises the following error
```python
from pyccel.stdlib.internal.blas import dgemv as core_dgemv

ERROR at annotation (semantic) stage
pyccel:
 |fatal [semantic]: utilities.py [14,4]| Undefined function (core_dgemv)
```

* **dnrm2** is not available, although it is declared in our header file
```shell
   51 |     r = dnrm2(n, x, incx)
      |        1
Error: Function ‘dnrm2’ at (1) has no IMPLICIT type;
```

* BLAS functions are not recognized, we are missing something like
```fortran
    real(f64), external :: ddot
```

# Summary

## Level 1

| SUBROUTINE  | PRECISIONS  | EXAMPLE     |
| ----------- | ----------- | ----------- |
| xROTG       | D           |             |
| xROTMG      | D           |             |
| xROT        | D           |             |
| xROTM       | D           |             |
| xSWAP       | D           |             |
| xSCAL       | D           |             |
| xCOPY       | D           |             |
| xAXPY       | D           |             |
| xDOT        | D           |             |
| xDOTU       |             |             |
| xDOTC       |             |             |
| xxDOT       |             |             |
| xNRM2       | D           |             |
| xASUM       | D           |             |
| IxAMAX      | D           |             |


## Level 2 

| SUBROUTINE  | PRECISIONS  | EXAMPLE     |
| ----------- | ----------- | ----------- |
| xGEMV       | D           |             |
| xGBMV       | D           | NOT WORKING |
| xHEMV       |             |             |
| xHBMV       |             |             |
| xHPMV       |             |             |
| xSYMV       | D           |             |
| xSBMV       | D           | NOT WORKING |
| xSPMV       |             |             |
| xTRMV       |             |             |
| xTBMV       |             |             |
| xTPMV       |             |             |
| xTRSV       |             |             |
| xTBSV       |             |             |
| xTPSV       |             |             |
| xGER        | D           |             |
| xGERU       |             |             |
| xGERC       |             |             |
| xHER        |             |             |
| xHPR        |             |             |
| xHER2       |             |             |
| xHPR2       |             |             |
| xSYR        |             |             |
| xSPR        |             |             |
| xSYR2       |             |             |
| xSPR2       |             |             |

## Level 3 

| SUBROUTINE  | PRECISIONS  | EXAMPLE     |
| ----------- | ----------- | ----------- |
| xGEMM       | D           |             |
| xSYMM       | D           |             |
| xHEMM       |             |             |
| xSYRK       | D           |             |
| xHERK       |             |             |
| xSYR2K      | D           |             |
| xHER2K      |             |             |
| xTRMM       |             |             |
| xTRSM       |             |             |
