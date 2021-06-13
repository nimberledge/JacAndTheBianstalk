from petsc4py import *
from petsc4py import PETSc
import graphviz
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

VERTEX_DEPTH_STRATUM = 0
EDGE_DEPTH_STRATUM = 1
FACE_DEPTH_STRATUM = 2

# Cell Quality Measure datatype
# We care about a cell's area (measure), minimum angle, aspect ratio, skewness, equiangle skew, and its scaled Jacobian
CQM = namedtuple('CQM', 'area, minAngle, aspectRatio, skewness, equiangleSkew, scaledJacobian')

def distance(p1, p2):
    if p1.shape[0] != p2.shape[0]:
        return ValueError
    
    return np.sqrt(np.sum([(p1[i] - p2[i])**2 for i in range(p1.shape[0])])) 

def getEdgeLength(plex, sec, dim, index):
    '''Takes in the index of an edge and outputs its length'''
    eStart, eEnd = plex.getDepthStratum(EDGE_DEPTH_STRATUM)
    assert eStart <= index < eEnd

    plexCoords = plex.getCoordinates()
    vertices = plex.getCone(index)
    v1, v2 = [np.array([plexCoords[sec.getOffset(v) + j] for j in range(dim)]) for v in vertices]
    return distance(v1, v2)

def getCellQualityMeasures(plex, sec, dim, index):
    '''Takes in the index of a cell and returns MeshQualityMeasures of the cell'''
    cStart, cEnd = plex.getDepthStratum(FACE_DEPTH_STRATUM)
    assert cStart <= index < cEnd
    
    plexCoords = plex.getCoordinates()
    edges = plex.getCone(index)
    vertices = set()
    minEdgeLength = np.inf
    maxEdgeLength = 0
    edgeLengths = []
    for e in edges:
        # Add vertices to a set (to avoid overcounting), use to calculate CQM
        verts = plex.getCone(e)
        v1, v2 = [([plexCoords.array[sec.getOffset(v) + j] for j in range(dim)]) for v in verts]
        vertices.add(tuple(v1))
        vertices.add(tuple(v2))
        
        # Compute edge length for aspect ratio and area calculations
        v1 = np.array(v1)
        v2 = np.array(v2)
        edgeLength = distance(v1, v2)
        edgeLengths.append(edgeLength)
        if edgeLength < minEdgeLength:
            minEdgeLength = edgeLength
        if edgeLength > maxEdgeLength:
            maxEdgeLength = edgeLength
        
    # Calculate area (Heron's formula) and aspect ratio from edge lengths
    # Not entirely sure, but I think we should be able to calculate skew as well from this
    edgeLengthRatio = maxEdgeLength / minEdgeLength
    aspectRatio = maxEdgeLength / minEdgeLength
    semiPerimeter = sum(edgeLengths) / 2
    area = semiPerimeter
    miniProduct = 1             # We are reusing this value = (s-a)(s-b)(s-c)
    edgeLengthsProduct = 1
    for i in range(len(edgeLengths)):
        miniProduct *= (semiPerimeter - edgeLengths[i])
        edgeLengthsProduct *= edgeLengths[i]
    area = np.sqrt(area * miniProduct)
    aspectRatio = edgeLengthsProduct / (8 * miniProduct)

    # Calculate angles at each vertex
    # If we know we're dealing with triangular meshes, I could just hard-code the vector calculations
    # I actually do not know how to write the code in general
    # To be fair, with triangles and tetrahedra every pair of vertices is connected
    v1, v2, v3 = [np.array(v) for v in vertices]
    # These guys are np arrays so i can do element-wise subtraction
    vec12 = v2 - v1
    vec23 = v3 - v2
    vec31 = v1 - v3
    dist12 = distance(v1, v2)
    dist23 = distance(v2, v3)
    dist31 = distance(v3, v1)
    a1 = np.arccos (np.dot(vec12, vec31) / (dist12 * dist31))
    a2 = np.arccos (np.dot(vec12, vec23) / (dist12 * dist23))
    a3 = np.arccos (np.dot(vec31, vec23) / (dist31 * dist23))
    minAngle = min(a1, a2, a3)
    maxAngle = max(a1, a2, a3)
    idealAngle = np.pi / 3      # There's gotta be a better way to write this stuff
    equiangleSkew = max( (maxAngle - idealAngle) / (np.pi - idealAngle), (idealAngle - minAngle) / idealAngle)

    # Calculating in accordance with https://www.engmorph.com/skewness-finite-elemnt
    # sideN -> side opposite vertex vN
    
    midPointSide1 = v2 + (v3 - v2) / 2
    midPointSide2 = v3 + (v1 - v3) / 2
    midPointSide3 = v1 + (v2 - v1) / 2
    # print ("Vertices: {} {}, midpoint: {}".format(v2, v3, midPointSide1))
    lineNormalSide1 = midPointSide1 - v1
    lineOrthSide1 = midPointSide3 - midPointSide2
    theta1 = np.arccos (np.dot(lineNormalSide1, lineOrthSide1) / (distance(v1, midPointSide1) * distance(midPointSide2, midPointSide3)))
    theta2 = np.pi - theta1

    lineNormalSide2 = midPointSide2 - v2
    lineOrthSide2 = midPointSide1 - midPointSide3
    theta3 = np.arccos (np.dot(lineNormalSide2, lineOrthSide2) / (distance(v2, midPointSide2) * distance(midPointSide1, midPointSide3)))
    theta4 = np.pi - theta3

    lineNormalSide3 = midPointSide3 - v3
    lineOrthSide3 = midPointSide2 - midPointSide1
    theta5 = np.arccos (np.dot(lineNormalSide3, lineOrthSide3) / (distance(v3, midPointSide3) * distance(midPointSide2, midPointSide1)))
    theta6 = np.pi - theta5

    skewness = (np.pi / 2) - min(theta1, theta2, theta3, theta4, theta5, theta6)
    scaledJacobian = 0

    return CQM(area, minAngle, aspectRatio, skewness, equiangleSkew, scaledJacobian)


def main():
    # Initial setup mesh
    dim = 2
    coords = np.asarray([
        [0.0, 0.0], # 0
        [0.5, 0.0], # 1
        [1.0, 0.0], # 2
        [0.0, 0.5], # 3
        [0.4, 0.6], # 4
        [1.0, 0.5], # 5
        [0.0, 1.0], # 6
        [0.5, 1.0], # 7
        [1.0, 1.0], # 8
    ], dtype=float)
    
    cells = np.asarray([
        [0, 1, 3],
        [1, 4, 3],
        [1, 2, 4],
        [2, 4, 5],
        [3, 4, 6],
        [4, 6, 7],
        [5, 7, 8],
        [4, 5, 7],
    ], dtype=PETSc.IntType)

    # Create DMPlex from cells and vertices
    plex = PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=PETSc.COMM_WORLD)
    # comm - the basic object used by MPI to determine which processes are involved in a communication 
    # PETSc.COMM_WORLD - the equivalent of the MPI_COMM_WORLD communicator which represents all the processes that PETSc knows about. 
    print (plex.getDimension())

    # Now, we set up a section so that we can smoothly translate between the co-ordinate plane and our DMPlex representation
    numComponents = 1
    entityDofs = [dim, 0, 0]  # 2 entries for each vertex, 0 for each edge, 0 for each cell
    plex.setNumFields(1)
    sec = plex.createSection(numComponents, entityDofs)
    sec.setFieldName(0, 'Coordinates')
    sec.setUp()
    plex.setSection(sec)

    # Some bookkeeping
    vStart, vEnd = plex.getDepthStratum(VERTEX_DEPTH_STRATUM)
    eStart, eEnd = plex.getDepthStratum(EDGE_DEPTH_STRATUM)
    cStart, cEnd = plex.getDepthStratum(FACE_DEPTH_STRATUM)
    
    # hStart, hEnd = plex.getDepthStratum(3)
    # print ("hStart, hEnd: {} {}".format(hStart, hEnd))
    
    Start, End = plex.getChart()
    plexCoords = plex.getCoordinates()

    # TEST FOR EDGE LENGTH FUNCTION
    # for edge in range(eStart, eEnd):
    #     try:
    #         edgeLength = getEdgeLength(plex, sec, dim, edge)
    #         print ("Edge: {}, Edge Length: {}".format(edge, edgeLength))
    #     except AssertionError:
    #         print ("{} is not an edge index. Skipped.".format(edge))
    
    # TEST FOR CELL QUALITY MEASURES
    for cell in range(cStart, cEnd):
        try:
            cellQuality = getCellQualityMeasures(plex, sec, dim, cell)
            print ("Cell: {} Quality: {}".format(cell, cellQuality))
        except AssertionError:
            print ("{} is not a cell index. Skipped.".format(cell))
            break

if __name__ == '__main__':
    main()