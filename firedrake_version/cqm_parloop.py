from firedrake import *

def computeArea(mesh, P0=None):
    '''Computes the area of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    areas = Function(P0)
    kernel = '''
    float x1 = coords[0];
    float y1 = coords[1];
    float x2 = coords[2];
    float y2 = coords[3];
    float x3 = coords[4];
    float y3 = coords[5];
    float d12 = sqrt (pow((x2 - x1), 2) + pow((y2 - y1), 2));
    float d23 = sqrt (pow((x3 - x2), 2) + pow((y3 - y2), 2));
    float d13 = sqrt (pow((x3 - x1), 2) + pow((y3 - y1), 2));
    float s = (d12 + d23 + d13) / 2;
    areas[0] = sqrt(s * (s - d12) * (s - d23) * (s - d13));
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'areas': (areas, RW)})
    return areas

def computeMinAngle(mesh, P0=None):
    '''Computes the minimum angle of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    minAngles = Function(P0)
    kernel = '''
    float x1 = coords[0];
    float y1 = coords[1];
    float x2 = coords[2];
    float y2 = coords[3];
    float x3 = coords[4];
    float y3 = coords[5];
    float x12 = x2 - x1;
    float y12 = y2 - y1;
    float x23 = x3 - x2;
    float y23 = y3 - y2;
    float x13 = x3 - x1;
    float y13 = y3 - y1;
    float d12 = sqrt (pow(x12, 2) + pow(y12, 2));
    float d23 = sqrt (pow(x23, 2) + pow(y23, 2));
    float d13 = sqrt (pow(x13, 2) + pow(y13, 2));
    float a1 = acos ((x12 * x13 + y12 * y13) / (d12 * d13));
    float a2 = acos (-1 * (x12 * x23 + y12 * y23) / (d12 * d23));
    float a3 = acos ((x23 * x13 + y23 * y13) / (d23 * d13));
    minAngles[0] = fmin(a1, a2);
    minAngles[0] = fmin(minAngles[0], a3);
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'minAngles': (minAngles, RW)})
    return minAngles

def computeAspectRatio(mesh, P0=None):
    '''Computes the aspect ratio of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    aspectRatios = Function(P0)
    kernel = '''
    float x1 = coords[0];
    float y1 = coords[1];
    float x2 = coords[2];
    float y2 = coords[3];
    float x3 = coords[4];
    float y3 = coords[5];
    float d12 = sqrt (pow((x2 - x1), 2) + pow((y2 - y1), 2));
    float d23 = sqrt (pow((x3 - x2), 2) + pow((y3 - y2), 2));
    float d13 = sqrt (pow((x3 - x1), 2) + pow((y3 - y1), 2));
    float s = (d12 + d23 + d13) / 2;
    aspectRatios[0] = (d12 * d23 * d13) / (8 * (s - d12) * (s - d23) * (s - d13));
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'aspectRatios': (aspectRatios, RW)})
    return aspectRatios

def computeEquiangleSkew(mesh, P0=None):
    '''Computes the equiangle skew of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    equiangleSkews = Function(P0)
    kernel = '''
    float x1 = coords[0];
    float y1 = coords[1];
    float x2 = coords[2];
    float y2 = coords[3];
    float x3 = coords[4];
    float y3 = coords[5];
    float x12 = x2 - x1;
    float y12 = y2 - y1;
    float x23 = x3 - x2;
    float y23 = y3 - y2;
    float x13 = x3 - x1;
    float y13 = y3 - y1;
    float d12 = sqrt (pow(x12, 2) + pow(y12, 2));
    float d23 = sqrt (pow(x23, 2) + pow(y23, 2));
    float d13 = sqrt (pow(x13, 2) + pow(y13, 2));
    float a1 = acos ((x12 * x13 + y12 * y13) / (d12 * d13));
    float a2 = acos (-1 * (x12 * x23 + y12 * y23) / (d12 * d23));
    float a3 = acos ((x23 * x13 + y23 * y13) / (d23 * d13));
    float minAngle = fmin(a1, a2);
    minAngle = fmin(minAngle, a3);
    float maxAngle = fmax(a1, a2);
    maxAngle = fmax(maxAngle, a3);
    float pi = 3.1415926535897;
    float idealAngle = pi / 3;
    skews[0] = fmax((maxAngle - idealAngle) / (pi - idealAngle), (idealAngle - minAngle) / idealAngle);
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'skews': (equiangleSkews, RW)})
    return equiangleSkews


def main():
    print ("Successful import")
    m,n = 4, 4
    mesh = UnitSquareMesh(m, n)
    areas = computeArea(mesh)
    # print (areas.dat.data)
    minAngles = computeMinAngle(mesh)
    # print (minAngles.dat.data)
    aspectRatios = computeAspectRatio(mesh)
    # print (aspectRatios.dat.data)
    equiangleSkews = computeEquiangleSkew(mesh)
    # print (equiangleSkews.dat.data)
    
if __name__ == '__main__':
    main()