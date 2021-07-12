from mqc import CQM, MeshQualityCalculator
from firedrake import *
from petsc4py import PETSc
import numpy as np

class TetrahedronMeshQualityCalculator(MeshQualityCalculator):

    def __init__(self, mesh):
        super().__init__(mesh)


def test_main():
    print ("Firedrake successfully imported")
    mesh = UnitCubeMesh(2, 2, 2)
    tmqc = TetrahedronMeshQualityCalculator(mesh)
    print (tmqc.meshType)
    cStart, cEnd = tmqc.getCellIndices()
    print (cStart, cEnd)
    for c in range(cStart, cEnd):
        print (tmqc.getCellQualityMeasures(c))

if __name__ == '__main__':
    test_main()
