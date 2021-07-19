from mqc import CQM, MeshQualityCalculator
from firedrake import *
from petsc4py import PETSc
import numpy as np

class TetrahedronMeshQualityCalculator(MeshQualityCalculator):

    def __init__(self, mesh):
        super().__init__(mesh)
    
    def getCellQualityMeasures(self, index):
        cStart, cEnd = self.plex.getDepthStratum(self.HEDRON_DEPTH_STRATUM)
        assert cStart <= index < cEnd
        
        faces = self.plex.getCone(index)
        edges = []
        for f in faces:
            edges.extend(self.plex.getCone(f))
        edges = list(set(edges))
        vertices = []
        for e in edges:
            vertices.extend(self.plex.getCone(e))
        vertices = list(set(vertices))
        vertices.sort()
        vertexCoords = np.array([np.array([self.coordArray[self.sec.getOffset(v) + j] \
         for j in range(self.dim)]) for v in vertices])

        v1 = vertexCoords[0, :]
        v2 = vertexCoords[1, :]
        v3 = vertexCoords[2, :]
        v4 = vertexCoords[3, :]

        vec12 = v2 - v1
        vec13 = v3 - v1
        vec14 = v4 - v1
        vec23 = v3 - v2
        vec24 = v4 - v2
        vec34 = v4 - v3

        dist12 = self.distance(v1, v2)
        dist13 = self.distance(v1, v3)
        dist14 = self.distance(v1, v4)
        dist23 = self.distance(v2, v3)
        dist24 = self.distance(v2, v4)
        dist34 = self.distance(v3, v4)

        volume = np.absolute(np.linalg.det(np.array([vec12, vec13, vec14])) / 6)

        # Calculate pointwise volume at the corners
        scaledJacobian1 = np.absolute(np.linalg.det([np.array([vec12, vec13, vec14])])) / (dist12 * dist13 * dist14)
        scaledJacobian2 = np.absolute(np.linalg.det([np.array([-vec12, vec23, vec24])])) / (dist12 * dist23 * dist24)
        scaledJacobian3 = np.absolute(np.linalg.det([np.array([-vec13, -vec23, vec34])])) / (dist13 * dist23 * dist34)
        scaledJacobian4 = np.absolute(np.linalg.det([np.array([-vec14, -vec24, -vec34])])) / (dist14 * dist24 * dist34)
        scaledJacobian = min(scaledJacobian1[0], scaledJacobian2[0], scaledJacobian3[0], scaledJacobian4[0])
        
        # https://en.wikipedia.org/wiki/Tetrahedron#Inradius - page describes inRadius and circumRadius calculations
        circumRadius = np.sqrt((dist12 * dist34 + dist13 * dist24 + dist14*dist23) * \
            (dist12 * dist34 + dist13 * dist24 - dist14*dist23) * \
            (dist12 * dist34 - dist13 * dist24 + dist14*dist23) * \
            (-dist12 * dist34 + dist13 * dist24 + dist14*dist23)) / (24 * volume)
        
        # faceAreaN = area of face opposite vertex N
        # edgeSumN = s in Heron's formula but for each face separately
        edgeSum1 = (dist23 + dist24 + dist34) / 2
        faceArea1 = np.sqrt(edgeSum1 * (edgeSum1 - dist23) * (edgeSum1 - dist24) * (edgeSum1 - dist34))
        edgeSum2 = (dist13 + dist14 + dist34) / 2       
        faceArea2 = np.sqrt(edgeSum2 * (edgeSum2 - dist13) * (edgeSum2 - dist14) * (edgeSum2 - dist34))
        edgeSum3 = (dist12 + dist14 + dist24) / 2       
        faceArea3 = np.sqrt(edgeSum3 * (edgeSum3 - dist12) * (edgeSum3 - dist14) * (edgeSum3 - dist24))
        edgeSum4 = (dist12 + dist23 + dist13) / 2       
        faceArea4 = np.sqrt(edgeSum4 * (edgeSum4 - dist12) * (edgeSum4 - dist23) * (edgeSum4 - dist13))
        
        inRadius = 3 * volume / (faceArea1 + faceArea2 + faceArea3 + faceArea4)

        # https://cubit.sandia.gov/15.5/help_manual/WebHelp/mesh_generation/mesh_quality_assessment/tetrahedral_metrics.htm
        # Link above describes the aspectRatioBeta which we take as our aspect ratio
        aspectRatio = circumRadius / (3 * inRadius)
        # angleij -> angle at vertex i, on the face opposite to vertex j
        # These are angles subtended by vertices at the faces of the hedron, each belonging to its own triangle
        angle12 = np.arccos(np.dot(vec13, vec14) / (dist13 * dist14))
        angle13 = np.arccos(np.dot(vec12, vec14) / (dist12 * dist14))
        angle14 = np.arccos(np.dot(vec12, vec13) / (dist12 * dist13))
        angle21 = np.arccos(np.dot(vec23, vec24) / (dist23 * dist24))
        angle23 = np.arccos(np.dot(-vec12, vec24) / (dist12 * dist24))
        angle24 = np.arccos(np.dot(-vec12, vec23) / (dist12 * dist23))
        angle31 = np.arccos(np.dot(-vec23, vec34) / (dist23 * dist34))
        angle32 = np.arccos(np.dot(-vec13, vec34) / (dist13 * dist34))
        angle34 = np.arccos(np.dot(-vec13, -vec23) / (dist13 * dist23))
        angle41 = np.arccos(np.dot(-vec24, -vec34) / (dist24 * dist34))
        angle42 = np.arccos(np.dot(-vec14, -vec34) / (dist14 * dist34))
        angle43 = np.arccos(np.dot(-vec14, -vec24) / (dist14 * dist24))
        
        minAngle = min(angle12, angle13, angle14, angle21, angle23, angle24, angle31, angle32, angle34, angle41, angle42, angle43)
        maxAngle = max(angle12, angle13, angle14, angle21, angle23, angle24, angle31, angle32, angle34, angle41, angle42, angle43)
        idealAngle = np.pi / 3         # This is still the ideal angle as the faces of a regular tetrahedron are regular triangles
        equiangleSkew = max( (maxAngle - idealAngle) / (np.pi - idealAngle),\
         (idealAngle - minAngle) / idealAngle)
        
        skewness = 0
        return CQM(volume, minAngle, aspectRatio, skewness, equiangleSkew, scaledJacobian)


def test_main():
    print ("Firedrake successfully imported")
    mesh = UnitCubeMesh(5, 5, 5)
    tmqc = TetrahedronMeshQualityCalculator(mesh)
    print ("Mesh type: {}".format(tmqc.meshType))
    cStart, cEnd = tmqc.getCellIndices()
    print ("cStart: {} cEnd: {}".format(cStart, cEnd))
    for c in range(cStart, cEnd):
        print (c, tmqc.getCellQualityMeasures(c))

if __name__ == '__main__':
    test_main()
