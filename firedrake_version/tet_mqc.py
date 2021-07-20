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
        
        # Calculating in accordance with https://www.engmorph.com/skewness-finite-elemnt
        # To extend to 3D case, drop a line from a vertex to opposite face centroid (line L)
        # Join midpoints of non-face edges, plane spanned by those is Q
        # Consider angles subtended by L at Q for skewness calculation

        # Calculate midpoints of sides
        midPoint12 = v1 + (vec12 / 2)
        midPoint13 = v1 + (vec13 / 2)
        midPoint14 = v1 + (vec14 / 2)
        midPoint23 = v2 + (vec23 / 2)
        midPoint24 = v2 + (vec24 / 2)
        midPoint34 = v3 + (vec34 / 2)

        # Calculate lines from vertices to face centroids
        faceCentroid1 = (v2 + v3 + v4) / 3
        faceCentroid2 = (v1 + v3 + v4) / 3
        faceCentroid3 = (v1 + v2 + v4) / 3
        faceCentroid4 = (v1 + v2 + v3) / 3
        vertexToFaceLine1 = faceCentroid1 - v1
        vertexToFaceLine2 = faceCentroid2 - v2
        vertexToFaceLine3 = faceCentroid3 - v3
        vertexToFaceLine4 = faceCentroid4 - v4

        # We calculate the normals of midpoint planes
        planeNormal1 = np.cross((midPoint12 - midPoint13), (midPoint12 - midPoint14))
        planeNormal2 = np.cross((midPoint12 - midPoint13), (midPoint12 - midPoint14))
        planeNormal3 = np.cross((midPoint12 - midPoint13), (midPoint12 - midPoint14))
        planeNormal4 = np.cross((midPoint12 - midPoint13), (midPoint12 - midPoint14))

        # Now, normalize to length = 1  and calculate angle
        # Distance of a vector to origin is the magnitude
        origin = np.zeros(v1.shape)
        planeNormal1 /= self.distance(planeNormal1, origin)
        planeNormal2 /= self.distance(planeNormal2, origin)
        planeNormal3 /= self.distance(planeNormal3, origin)
        planeNormal4 /= self.distance(planeNormal4, origin)
        vertexToFaceLine1 /= self.distance(vertexToFaceLine1, origin)
        vertexToFaceLine2 /= self.distance(vertexToFaceLine2, origin)
        vertexToFaceLine3 /= self.distance(vertexToFaceLine3, origin)
        vertexToFaceLine4 /= self.distance(vertexToFaceLine4, origin)

        # This is the angle between the normal and the line. We need pi/2 - angle
        # as the angle subtended at the plane
        # Unless angle > pi/2 then we need angle - pi/2
        theta1 = np.arccos(np.dot(planeNormal1, vertexToFaceLine1))
        if theta1 >= np.pi / 2:
            theta1 = theta1 - np.pi / 2
        else:
            theta1 = np.pi / 2 - theta1
        theta2 = np.pi - theta1

        theta3 = np.arccos(np.dot(planeNormal2, vertexToFaceLine2))
        if theta3 >= np.pi / 2:
            theta3 = theta3 - np.pi / 2
        else:
            theta3 = np.pi / 2 - theta3
        theta4 = np.pi - theta3
        
        theta5 = np.arccos(np.dot(planeNormal3, vertexToFaceLine3))
        if theta5 >= np.pi / 2:
            theta5 = theta5 - np.pi / 2
        else:
            theta5 = np.pi / 2 - theta5
        theta6 = np.pi - theta5

        theta7 = np.arccos(np.dot(planeNormal4, vertexToFaceLine4))
        if theta7 >= np.pi / 2:
            theta7 = theta7 - np.pi / 2
        else:
            theta7 = np.pi / 2 - theta7
        theta8 = np.pi - theta7
        
        skewness = (np.pi / 2) - min(theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8)
        
        return CQM(volume, minAngle, aspectRatio, skewness, equiangleSkew, scaledJacobian)


def test_main():
    mesh = UnitCubeMesh(2, 2, 2)
    tmqc = TetrahedronMeshQualityCalculator(mesh)
    print ("Mesh type: {}".format(tmqc.meshType))
    cStart, cEnd = tmqc.getCellIndices()
    print ("cStart: {} cEnd: {}".format(cStart, cEnd))
    volumeSum = 0
    for c in range(cStart, cEnd):
        cqm = tmqc.getCellQualityMeasures(c)
        print ("{}\t{}".format(c, tmqc.printCQM(cqm)))
        volumeSum += cqm.measure
    
    # print ("Volume sum: {}".format(volumeSum))
    tolerance = 10**(-6)
    assert np.absolute(volumeSum - 1) < tolerance

if __name__ == '__main__':
    test_main()
