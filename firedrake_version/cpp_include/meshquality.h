#include <iostream>
#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector2d p1, Vector2d p2) {
    return sqrt ( pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) );
}

void getMetric(double *metrics, const double *T_, double *coords) {
    // Map vertices as vectors
    Map<Vector2d> V1((double *) &coords[0]);
    Map<Vector2d> V2((double *) &coords[2]);
    Map<Vector2d> V3((double *) &coords[4]);
    
    // Precompute some vectors, and distances
    Vector2d V12 = V2 - V1;
    Vector2d V13 = V3 - V1;
    Vector2d V23 = V3 - V2;
    double d12 = distance(V1, V2);
    double d13 = distance(V1, V3);
    double d23 = distance(V2, V3);
    double s = (d12 + d23 + d13) / 2;
    double area = sqrt(s * (s-d12) * (s-d13) * (s-d23));

    // Map tensor as 2x2 Matrices
    Map<Matrix2d> M1((double *) &T_[0]);
    Map<Matrix2d> M2((double *) &T_[4]);
    Map<Matrix2d> M3((double *) &T_[8]);

    // Compute M(x, y) at centroid x_c to get area_M
    Matrix2d Mxc = (M1 + M2 + M3) / 3;
    double areaM = area * sqrt(Mxc.determinant());
    
    // Compute edge lengths in metric
    double L1 = V23.dot(((M2 + M3)/2) * V23);
    double L2 = V13.dot(((M1 + M3)/2) * V13);
    double L3 = V12.dot(((M1 + M2)/2) * V12);

    metrics[0] = sqrt(3) * (L1 + L2 + L3) / (2 * areaM);
}