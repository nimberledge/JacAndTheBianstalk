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

def computeScaledJacobian(mesh, P0=None):
    '''Computes the scaled Jacobian of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    scaledJacobians = Function(P0)
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
    float sj1 = fabs(x12 * y13 - x13 * y12) / (d12 * d13);
    float sj2 = fabs(x12 * y23 - x23 * y12) / (d12 * d23);
    float sj3 = fabs(x23 * y13 - x13 * y23) / (d23 * d13);
    scaledJacobians[0] = fmin(sj1, sj2);
    scaledJacobians[0] = fmin(scaledJacobians[0], sj3);
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'scaledJacobians': (scaledJacobians, RW)})
    return scaledJacobians

def computeSkewness(mesh, P0=None):
    '''Computes the scaled Jacobian of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    skews = Function(P0)
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
    
    float mx1 = x2 + x23/2;
    float my1 = y2 + y23/2;
    float mx2 = x1 + x13/2;
    float my2 = y1 + y13/2;
    float mx3 = x1 + x12/2;
    float my3 = y1 + y12/2;

    float dpm11 = sqrt (pow((x1 - mx1), 2) +  pow((y1 - my1), 2) );
    float dpm22 = sqrt (pow((x2 - mx2), 2) +  pow((y2 - my2), 2) );
    float dpm33 = sqrt (pow((x3 - mx3), 2) +  pow((y3 - my3), 2) );
    
    float dm12 = sqrt ( pow((mx1 - mx2), 2) + pow((my1 - my2), 2) );
    float dm23 = sqrt ( pow((mx3 - mx2), 2) + pow((my3 - my2), 2) );
    float dm13 = sqrt ( pow((mx1 - mx3), 2) + pow((my1 - my3), 2) );

    float lnx1 = mx1 - x1;
    float lny1 = my1 - y1;
    float lox1 = mx3 - mx2;
    float loy1 = my3 - my2;

    float lnx2 = mx2 - x2;
    float lny2 = my2 - y2;
    float lox2 = mx1 - mx3;
    float loy2 = my1 - my3;

    float lnx3 = mx3 - x3;
    float lny3 = my3 - y3;
    float lox3 = mx2 - mx1;
    float loy3 = my2 - my1;

    float pi = 3.1415926535897;
    float min_t = 0.0;
    float t1 = acos ((lnx1 * lox1 + lny1 * loy1) / (dpm11 * dm23));
    float t2 = pi - t1;
    min_t = fmin(t1, t2);

    float t3 = acos ((lnx2 * lox2 + lny2 * loy2) / (dpm22 * dm13));
    min_t = fmin(min_t, t3);
    float t4 = pi - t3;
    min_t = fmin(min_t, t4);

    float t5 = acos ((lnx3 * lox3 + lny3 * loy3) / (dpm33 * dm12));
    min_t = fmin(min_t, t5);
    float t6 = pi - t5;
    min_t = fmin(min_t, t6);
    
    skews[0] = pi/2 - min_t;
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'skews': (skews, RW)})
    return skews

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
    scaledJacobians = computeScaledJacobian(mesh)
    # print (scaledJacobians.dat.data)
    skews = computeSkewness(mesh)
    # print (skews.dat.data)
    
if __name__ == '__main__':
    main()