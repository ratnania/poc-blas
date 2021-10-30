# BUGS

* this import raises the following error
```python
from pyccel.stdlib.internal.blas import dgemv as core_dgemv

ERROR at annotation (semantic) stage
pyccel:
 |fatal [semantic]: utilities.py [14,4]| Undefined function (core_dgemv)
```


# Summary

## Level 1

| SUBROUTINE  | PRECISIONS  | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE) |    (DONE)   |             |
| ----------- | ----------- | ----------- | ----------- |
| xROTG       | S,D         | SD          |             |
| xROTMG      | S,D         | SD          |             |
| xROT        | S,D         | SD          |             |
| xROTM       | S,D         | SD          |             |
| xSWAP       | S,D,C,Z     | SDC         |             |
| xSCAL       | S,D,C,Z     | SDC         |             |
| xCOPY       | S,D,C,Z     | SDC         |             |
| xAXPY       | S,D,C,Z     | SDC         |             |
| xDOT        | S,D-DS      | SD          |             |
| xDOTU       | C,Z         | C           | KO -> C     |
| xDOTC       | C,Z         | C           | KO -> C     |
| xxDOT       | SDS         |             |             |
| xNRM2       | S,D,SC,DZ   | SDC         |             |
| xASUM       | S,D,SC,DZ   | SDC         | KO -> S,C   |
| IxAMAX      | S,D,C,Z     | SDc         |             |


## Level 2 

| SUBROUTINE  | PRECISIONS   | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE)  |   (DONE)    |             |
| ----------- | -----------  | ----------- | ----------- |
| xGEMV       | S,D,C,Z      | SDC         |             |
| xGBMV       | S,D,C,Z      | SDC         |             |
| xHEMV       | C,Z          | C           |             |
| xHBMV       | C,Z          | C           |             |
| xHPMV       | C,Z          |             |             |
| xSYMV       | S,D          | SD          |             |
| xSBMV       | S,D          | SD          |             |
| xSPMV       | S,D          | SD          |             |
| xTRMV       | S,D,C,Z      | SDC         |             |
| xTBMV       | S,D,C,Z      | SDC         |             |
| xTPMV       | S,D,C,Z      | SDC         |             |
| xTRSV       | S,D,C,Z      | SDC         |             |
| xTBSV       | S,D,C,Z      | SDC         |             |
| xTPSV       | S,D,C,Z      | SDC         |             |
| xGER        | S,D          | SD          |             |
| xGERU       | C,Z          |             |             |
| xGERC       | C,Z          |             |             |
| xHER        | C,Z          |             |             |
| xHPR        | C,Z          |             |             |
| xHER2       | C,Z          |             |             |
| xHPR2       | C,Z          |             |             |
| xSYR        | S,D          | SD          |             |
| xSPR        | S,D          | SD          |             |
| xSYR2       | S,D          | SD          |             |
| xSPR2       | S,D          | SD          |             |

## Level 3 

| SUBROUTINE  | PRECISIONS   | PRECISIONS  | EXAMPLE     |
|             | (AVAILABLE)  |   (DONE)    |             |
| ----------- | -----------  | ----------- | ----------- |
| xGEMM       | S,D,C,Z      | SDC         |             |
| xSYMM       | S,D,C,Z      | SDC         |             |
| xHEMM       | C,Z          |             |             |
| xSYRK       | S,D,C,Z      | SDC         |             |
| xHERK       | C,Z          |             |             |
| xSYR2K      | S,D,C,Z      | SDC         |             |
| xHER2K      | C,Z          |             |             |
| xTRMM       | S,D,C,Z      | SDC         |             |
| xTRSM       | S,D,C,Z      | SDC         |             |
