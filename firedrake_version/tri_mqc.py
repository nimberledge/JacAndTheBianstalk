from mqc import CQM, MeshQualityCalculator
from firedrake import *
from petsc4py import PETSc
import numpy as np

class TriangleMeshQualityCalculator(MeshQualityCalculator):

    def __init__(self, mesh):
        super().__init__(mesh)

    def getCellQualityMeasures(self, index):
        cStart, cEnd = self.plex.getDepthStratum(self.FACE_DEPTH_STRATUM)
        assert cStart <= index < cEnd

        edges = self.plex.getCone(index)
        vertices = set()
        for e in edges:
            v1, v2 = self.plex.getCone(e)
            vertices.add(v1)
            vertices.add(v2)
        vertices = list(vertices)
        vertices.sort()
        vertexCoords = np.array([np.array([self.coordArray[self.sec.getOffset(v) + j] \
         for j in range(self.dim)]) for v in vertices])

        # Re-represent the vertices for geometry calculations
        v1 = vertexCoords[0, :]
        v2 = vertexCoords[1, :]
        v3 = vertexCoords[2, :]

        vec12 = v2 - v1
        vec13 = v3 - v1
        vec23 = v3 - v2
        dist12 = self.distance(v1, v2)
        dist13 = self.distance(v1, v3)
        dist23 = self.distance(v2, v3)
        edgeLengths = [dist12, dist13, dist23]

        # Get scaled jacobian
        # https://cubit.sandia.gov/15.5/help_manual/WebHelp/mesh_generation/mesh_quality_assessment/triangular_metrics.htm
        # https://www.osti.gov/biblio/5009 - shows how to calculate the jacobian
        scaledJacobian1 = np.absolute(np.linalg.det(np.array([vec12, vec13]))) / (dist12 * dist13)
        scaledJacobian2 = np.absolute(np.linalg.det(np.array([vec23, -vec12]))) / (dist23 * dist12)
        scaledJacobian3 = np.absolute(np.linalg.det(np.array([-vec13, -vec23]))) / (dist13 * dist23)
        scaledJacobian = min(scaledJacobian1, scaledJacobian2, scaledJacobian3)

        # Get angles at vertices
        a1 = np.arccos(np.dot(vec12, vec13) / (dist12 * dist13))
        a2 = np.arccos(np.dot(vec12, -vec23) / (dist12 * dist23))
        a3 = np.arccos(np.dot(-vec13, -vec23) / (dist13 * dist23))
        minAngle = min(a1, a2, a3)
        maxAngle = max(a1, a2, a3)

        semiPerimeter = sum(edgeLengths) / 2
        area = semiPerimeter
        miniProduct = 1             # (s-a) * (s-b) * (s-c)
        edgeLengthsProduct = 1
        for i in range(len(edgeLengths)):
            miniProduct *= (semiPerimeter - edgeLengths[i])
            edgeLengthsProduct *= edgeLengths[i]
        area = np.sqrt(area * miniProduct)
        aspectRatio = edgeLengthsProduct / (8 * miniProduct)

        idealAngle = np.pi / 3
        equiangleSkew = max( (maxAngle - idealAngle) / (np.pi - idealAngle),\
         (idealAngle - minAngle) / idealAngle)

        midPointSide1 = v2 + (v3 - v2) / 2
        midPointSide2 = v3 + (v1 - v3) / 2
        midPointSide3 = v1 + (v2 - v1) / 2

        lineNormalSide1 = midPointSide1 - v1
        lineOrthSide1 = midPointSide3 - midPointSide2
        theta1 = np.arccos (np.dot(lineNormalSide1, lineOrthSide1) / (self.distance(v1, midPointSide1) * self.distance(midPointSide2, midPointSide3)))
        theta2 = np.pi - theta1

        lineNormalSide2 = midPointSide2 - v2
        lineOrthSide2 = midPointSide1 - midPointSide3
        theta3 = np.arccos (np.dot(lineNormalSide2, lineOrthSide2) / (self.distance(v2, midPointSide2) * self.distance(midPointSide1, midPointSide3)))
        theta4 = np.pi - theta3

        lineNormalSide3 = midPointSide3 - v3
        lineOrthSide3 = midPointSide2 - midPointSide1
        theta5 = np.arccos (np.dot(lineNormalSide3, lineOrthSide3) / (self.distance(v3, midPointSide3) * self.distance(midPointSide2, midPointSide1)))
        theta6 = np.pi - theta5

        skewness = (np.pi / 2) - min(theta1, theta2, theta3, theta4, theta5, theta6)
        return CQM(area, minAngle, aspectRatio, skewness, equiangleSkew, scaledJacobian)

def test_main():
    print ("Firedrake successfully imported")
    mesh = UnitSquareMesh(5, 5)
    tmqc = TriangleMeshQualityCalculator(mesh)
    print (tmqc.meshType)
    cStart, cEnd = tmqc.getCellIndices()
    for c in range(cStart, cEnd):
        print (tmqc.getCellQualityMeasures(c))

if __name__ == '__main__':
    test_main()
