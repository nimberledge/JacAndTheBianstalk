from firedrake import *
import time

def computeArea(mesh, P0=None):
    '''Computes the area of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    areas = Function(P0)
    kernel = '''
    double x1 = coords[0];
    double y1 = coords[1];
    double x2 = coords[2];
    double y2 = coords[3];
    double x3 = coords[4];
    double y3 = coords[5];
    double d12 = sqrt (pow((x2 - x1), 2) + pow((y2 - y1), 2));
    double d23 = sqrt (pow((x3 - x2), 2) + pow((y3 - y2), 2));
    double d13 = sqrt (pow((x3 - x1), 2) + pow((y3 - y1), 2));
    double s = (d12 + d23 + d13) / 2;
    areas[0] = sqrt(s * (s - d12) * (s - d23) * (s - d13));
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'areas': (areas, RW)})
    return areas

def computeMinAngle(mesh, P0=None):
    '''Computes the minimum angle of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    minAngles = Function(P0)
    kernel = '''
    double x1 = coords[0];
    double y1 = coords[1];
    double x2 = coords[2];
    double y2 = coords[3];
    double x3 = coords[4];
    double y3 = coords[5];
    double x12 = x2 - x1;
    double y12 = y2 - y1;
    double x23 = x3 - x2;
    double y23 = y3 - y2;
    double x13 = x3 - x1;
    double y13 = y3 - y1;
    double d12 = sqrt (pow(x12, 2) + pow(y12, 2));
    double d23 = sqrt (pow(x23, 2) + pow(y23, 2));
    double d13 = sqrt (pow(x13, 2) + pow(y13, 2));
    double a1 = acos ((x12 * x13 + y12 * y13) / (d12 * d13));
    double a2 = acos (-1 * (x12 * x23 + y12 * y23) / (d12 * d23));
    double a3 = acos ((x23 * x13 + y23 * y13) / (d23 * d13));
    minAngles[0] = fmin(a1, a2);
    minAngles[0] = fmin(minAngles[0], a3);
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'minAngles': (minAngles, RW)})
    return minAngles

def computeAspectRatio(mesh, P0=None):
    '''Computes the aspect ratio of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    aspectRatios = Function(P0)
    kernel = '''
    double x1 = coords[0];
    double y1 = coords[1];
    double x2 = coords[2];
    double y2 = coords[3];
    double x3 = coords[4];
    double y3 = coords[5];
    double d12 = sqrt (pow((x2 - x1), 2) + pow((y2 - y1), 2));
    double d23 = sqrt (pow((x3 - x2), 2) + pow((y3 - y2), 2));
    double d13 = sqrt (pow((x3 - x1), 2) + pow((y3 - y1), 2));
    double s = (d12 + d23 + d13) / 2;
    aspectRatios[0] = (d12 * d23 * d13) / (8 * (s - d12) * (s - d23) * (s - d13));
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'aspectRatios': (aspectRatios, RW)})
    return aspectRatios

def computeEquiangleSkew(mesh, P0=None):
    '''Computes the equiangle skew of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    equiangleSkews = Function(P0)
    kernel = '''
    double x1 = coords[0];
    double y1 = coords[1];
    double x2 = coords[2];
    double y2 = coords[3];
    double x3 = coords[4];
    double y3 = coords[5];
    double x12 = x2 - x1;
    double y12 = y2 - y1;
    double x23 = x3 - x2;
    double y23 = y3 - y2;
    double x13 = x3 - x1;
    double y13 = y3 - y1;
    double d12 = sqrt (pow(x12, 2) + pow(y12, 2));
    double d23 = sqrt (pow(x23, 2) + pow(y23, 2));
    double d13 = sqrt (pow(x13, 2) + pow(y13, 2));
    double a1 = acos ((x12 * x13 + y12 * y13) / (d12 * d13));
    double a2 = acos (-1 * (x12 * x23 + y12 * y23) / (d12 * d23));
    double a3 = acos ((x23 * x13 + y23 * y13) / (d23 * d13));
    double minAngle = fmin(a1, a2);
    minAngle = fmin(minAngle, a3);
    double maxAngle = fmax(a1, a2);
    maxAngle = fmax(maxAngle, a3);
    double pi = 3.1415926535897;
    double idealAngle = pi / 3;
    skews[0] = fmax((maxAngle - idealAngle) / (pi - idealAngle), (idealAngle - minAngle) / idealAngle);
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'skews': (equiangleSkews, RW)})
    return equiangleSkews

def computeScaledJacobian(mesh, P0=None):
    '''Computes the scaled Jacobian of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    scaledJacobians = Function(P0)
    kernel = '''
    double x1 = coords[0];
    double y1 = coords[1];
    double x2 = coords[2];
    double y2 = coords[3];
    double x3 = coords[4];
    double y3 = coords[5];
    double x12 = x2 - x1;
    double y12 = y2 - y1;
    double x23 = x3 - x2;
    double y23 = y3 - y2;
    double x13 = x3 - x1;
    double y13 = y3 - y1;
    double d12 = sqrt (pow(x12, 2) + pow(y12, 2));
    double d23 = sqrt (pow(x23, 2) + pow(y23, 2));
    double d13 = sqrt (pow(x13, 2) + pow(y13, 2));
    double sj1 = fabs(x12 * y13 - x13 * y12) / (d12 * d13);
    double sj2 = fabs(x12 * y23 - x23 * y12) / (d12 * d23);
    double sj3 = fabs(x23 * y13 - x13 * y23) / (d23 * d13);
    scaledJacobians[0] = fmin(sj1, sj2);
    scaledJacobians[0] = fmin(scaledJacobians[0], sj3);
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'scaledJacobians': (scaledJacobians, RW)})
    return scaledJacobians

def computeSkewness(mesh, P0=None):
    '''Computes the skewness of each element of a triangular 2D mesh using a C kernel'''
    if P0 is None:
        P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    skews = Function(P0)
    kernel = '''
    double x1 = coords[0];
    double y1 = coords[1];
    double x2 = coords[2];
    double y2 = coords[3];
    double x3 = coords[4];
    double y3 = coords[5];
    double x12 = x2 - x1;
    double y12 = y2 - y1;
    double x23 = x3 - x2;
    double y23 = y3 - y2;
    double x13 = x3 - x1;
    double y13 = y3 - y1;
    
    double mx1 = x2 + x23/2;
    double my1 = y2 + y23/2;
    double mx2 = x1 + x13/2;
    double my2 = y1 + y13/2;
    double mx3 = x1 + x12/2;
    double my3 = y1 + y12/2;

    double dpm1 = sqrt (pow((x1 - mx1), 2) +  pow((y1 - my1), 2) );
    double dpm2 = sqrt (pow((x2 - mx2), 2) +  pow((y2 - my2), 2) );
    double dpm3 = sqrt (pow((x3 - mx3), 2) +  pow((y3 - my3), 2) );
    
    double dm12 = sqrt ( pow((mx1 - mx2), 2) + pow((my1 - my2), 2) );
    double dm23 = sqrt ( pow((mx3 - mx2), 2) + pow((my3 - my2), 2) );
    double dm13 = sqrt ( pow((mx1 - mx3), 2) + pow((my1 - my3), 2) );

    double lnx1 = mx1 - x1;
    double lny1 = my1 - y1;
    double lox1 = mx3 - mx2;
    double loy1 = my3 - my2;

    double lnx2 = mx2 - x2;
    double lny2 = my2 - y2;
    double lox2 = mx1 - mx3;
    double loy2 = my1 - my3;

    double lnx3 = mx3 - x3;
    double lny3 = my3 - y3;
    double lox3 = mx2 - mx1;
    double loy3 = my2 - my1;

    double pi = 3.1415926535897;
    double min_t = 0.0;
    double t1 = acos ((lnx1 * lox1 + lny1 * loy1) / (dpm1 * dm23));
    double t2 = pi - t1;
    min_t = fmin(t1, t2);

    double t3 = acos ((lnx2 * lox2 + lny2 * loy2) / (dpm2 * dm13));
    min_t = fmin(min_t, t3);
    double t4 = pi - t3;
    min_t = fmin(min_t, t4);

    double t5 = acos ((lnx3 * lox3 + lny3 * loy3) / (dpm3 * dm12));
    min_t = fmin(min_t, t5);
    double t6 = pi - t5;
    min_t = fmin(min_t, t6);
    
    skews[0] = pi/2 - min_t;
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'skews': (skews, RW)})
    return skews

def getCQM(mesh, P0=None):
    '''Computes the CQM (Cell Quality Measures) of each element of a triangular 2D mesh using a C kernel.
    CQMs are as follows - 
    1. Area
    2. Minimum Angle
    3. Aspect Ratio
    4. Equiangle Skew
    5. Skewness
    6. Scaled Jacobian'''
    if P0 is None:
        P0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    
    coords = mesh.coordinates
    areas = Function(P0)
    minAngles = Function(P0)
    aspectRatios = Function(P0)
    eSkews = Function(P0)
    skews = Function(P0)
    scaledJacobians = Function(P0)
    kernel = '''
    double x1 = coords[0];
    double y1 = coords[1];
    double x2 = coords[2];
    double y2 = coords[3];
    double x3 = coords[4];
    double y3 = coords[5];
    double x12 = x2 - x1;
    double y12 = y2 - y1;
    double x23 = x3 - x2;
    double y23 = y3 - y2;
    double x13 = x3 - x1;
    double y13 = y3 - y1;
    double d12 = sqrt (pow(x12, 2) + pow(y12, 2));
    double d23 = sqrt (pow(x23, 2) + pow(y23, 2));
    double d13 = sqrt (pow(x13, 2) + pow(y13, 2));
    double s = (d12 + d23 + d13) / 2;
    double pi = 3.14159265358979323846;
    
    // Scaled Jacobians
    double sj1 = fabs(x12 * y13 - x13 * y12) / (d12 * d13);
    double sj2 = fabs(x12 * y23 - x23 * y12) / (d12 * d23);
    double sj3 = fabs(x23 * y13 - x13 * y23) / (d23 * d13);
    scaledJacobians[0] = fmin(sj1, sj2);
    scaledJacobians[0] = fmin(scaledJacobians[0], sj3);

    // Minimum Angles
    double a1 = acos ((x12 * x13 + y12 * y13) / (d12 * d13));
    double a2 = acos (-1 * (x12 * x23 + y12 * y23) / (d12 * d23));
    double a3 = acos ((x23 * x13 + y23 * y13) / (d23 * d13));
    minAngles[0] = fmin(a1, a2);
    minAngles[0] = fmin(minAngles[0], a3);

    // Equiangle Skews
    double maxAngle = fmax(a1, a2);
    maxAngle = fmax(maxAngle, a3);
    double idealAngle = pi / 3;
    eSkews[0] = fmax((maxAngle - idealAngle) / (pi - idealAngle), (idealAngle - minAngles[0]) / idealAngle);

    // Areas
    areas[0] = sqrt(s * (s - d12) * (s - d23) * (s - d13));

    // Aspect Ratios
    aspectRatios[0] = (d12 * d23 * d13) / (8 * (s - d12) * (s - d23) * (s - d13));

    // Skewness
    // Refer Python file for variable names and calculation reference
    double mx1 = x2 + x23/2;
    double my1 = y2 + y23/2;
    double mx2 = x1 + x13/2;
    double my2 = y1 + y13/2;
    double mx3 = x1 + x12/2;
    double my3 = y1 + y12/2;

    double dpm1 = sqrt (pow((x1 - mx1), 2) +  pow((y1 - my1), 2) );
    double dpm2 = sqrt (pow((x2 - mx2), 2) +  pow((y2 - my2), 2) );
    double dpm3 = sqrt (pow((x3 - mx3), 2) +  pow((y3 - my3), 2) );
    
    double dm12 = sqrt ( pow((mx1 - mx2), 2) + pow((my1 - my2), 2) );
    double dm23 = sqrt ( pow((mx3 - mx2), 2) + pow((my3 - my2), 2) );
    double dm13 = sqrt ( pow((mx1 - mx3), 2) + pow((my1 - my3), 2) );

    double lnx1 = mx1 - x1;
    double lny1 = my1 - y1;
    double lox1 = mx3 - mx2;
    double loy1 = my3 - my2;

    double lnx2 = mx2 - x2;
    double lny2 = my2 - y2;
    double lox2 = mx1 - mx3;
    double loy2 = my1 - my3;

    double lnx3 = mx3 - x3;
    double lny3 = my3 - y3;
    double lox3 = mx2 - mx1;
    double loy3 = my2 - my1;

    double min_t = 0.0;
    double t1 = acos ((lnx1 * lox1 + lny1 * loy1) / (dpm1 * dm23));
    double t2 = pi - t1;
    min_t = fmin(t1, t2);

    double t3 = acos ((lnx2 * lox2 + lny2 * loy2) / (dpm2 * dm13));
    min_t = fmin(min_t, t3);
    double t4 = pi - t3;
    min_t = fmin(min_t, t4);

    double t5 = acos ((lnx3 * lox3 + lny3 * loy3) / (dpm3 * dm12));
    min_t = fmin(min_t, t5);
    double t6 = pi - t5;
    min_t = fmin(min_t, t6);
    
    skews[0] = pi/2 - min_t;
    '''
    par_loop(kernel, dx, {'coords': (coords, READ), 'skews': (skews, RW), \
                          'areas': (areas, RW), 'scaledJacobians': (scaledJacobians, RW), \
                          'minAngles': (minAngles, RW), 'aspectRatios': (aspectRatios, RW), \
                          'eSkews': (eSkews, RW)})
    return (areas, minAngles, aspectRatios, eSkews, skews, scaledJacobians)

def main():
    m,n = 4, 4
    mesh = UnitSquareMesh(m, n)
    # areas = computeArea(mesh)
    # minAngles = computeMinAngle(mesh)
    # aspectRatios = computeAspectRatio(mesh)
    # equiangleSkews = computeEquiangleSkew(mesh)
    # scaledJacobians = computeScaledJacobian(mesh)
    # skews = computeSkewness(mesh)

    start = time.time()
    areas, minAngles, aspectRatios, equiangleSkews, skews, scaledJacobians = getCQM(mesh)
    timeTaken = time.time() - start
    cqms = np.zeros((areas.dat.data.shape[0], 6))
    
    cqms[:, 0] = areas.dat.data
    cqms[:, 1] = minAngles.dat.data
    cqms[:, 2] = aspectRatios.dat.data
    cqms[:, 3] = skews.dat.data
    cqms[:, 4] = equiangleSkews.dat.data
    cqms[:, 5] = scaledJacobians.dat.data
    
    print ("Mesh size: {} x {}".format(m, n))
    print ("Number of cells: {}".format(areas.dat.data.shape[0]))
    print ("Area\t\tMin Angle\tAspect Ratio\tSkewness\tEq. skew\tScaled Jacobian")
    for r in range(cqms.shape[0]):
        print ('\t'.join(["{:.6f}".format(k) for k in cqms[r, :]]))
    print ("Time taken: {}s".format(timeTaken))
    
if __name__ == '__main__':
    main()