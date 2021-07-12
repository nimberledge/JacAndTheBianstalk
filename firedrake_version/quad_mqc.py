from mqc import CQM, MeshQualityCalculator
from firedrake import *
from petsc4py import PETSc
import numpy as np

class QuadrilateralMeshQualityCalculator(MeshQualityCalculator):

    def __init__(self, mesh):
        super().__init__(mesh)


def test_main():
    print ("Firedrake successfully imported")
    mesh = UnitSquareMesh(2, 2, quadrilateral=True)
    qmqc = QuadrilateralMeshQualityCalculator(mesh)
    print (qmqc.meshType)
    cStart, cEnd = qmqc.getCellIndices()
    print (cStart, cEnd)
    # for c in range(cStart, cEnd):
    #     print (qmqc.getCellQualityMeasures(c))

if __name__ == '__main__':
    test_main()
