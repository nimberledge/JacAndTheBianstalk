from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def setup_section_coords(plex, dim=2):
    numComponents = 1
    entityDofs = [dim, 0, 0]
    plex.setNumFields(numComponents)
    sec = plex.createSection(numComponents, entityDofs)
    sec.setFieldName(0, 'Coordinates')
    sec.setUp()
    plex.setSection(sec)
    return sec

def visualize_2dmesh(mesh, fig=None, dim=2):
    plex = mesh._topology.topology_dm
    plexCoords = plex.getCoordinates()
    sec = setup_section_coords(plex)
    vStart, vEnd = plex.getDepthStratum(0)
    eStart, eEnd = plex.getDepthStratum(1)
    if not fig:
        fig = plt.figure()

    ax = fig.add_subplot()
    for v in range(vStart, vEnd):
        vCoords = ([plexCoords.array[sec.getOffset(v) + j] for j in range(dim)])
        plt.plot(vCoords[0], vCoords[1], 'bo')
        ax.annotate(str(v), xy=vCoords)

    for e in range(eStart, eEnd):
        vertCoords = np.array([([plexCoords.array[sec.getOffset(v) + j] for j in range(dim)]) for v in plex.getCone(e)])
        plt.plot(vertCoords[:, 0], vertCoords[:, 1], 'k-')

    return ax

def test_main():
    print ("Firedrake successfully imported")
    mesh = UnitSquareMesh(2, 2)
    dim = 2
    fig = visualize_2dmesh(mesh)
    plt.show()



if __name__ == '__main__':
    test_main()
