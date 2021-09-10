#include <iostream>
#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;

#define PI 3.14159265358979323846

double distance(Vector2d p1, Vector2d p2) {
    return sqrt ( pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) );
}

void getCQM(double *areas, double *minAngles, double *aspectRatios, double *eSkews,
            double *skews, double *scaledJacobians, double *coords) {    
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


    // Scaled Jacobian
    // TODO: Maybe refactor this into having a determinant calculation...
    // Stuff that in another function so that I can use it for tets
    double sj1 = std::abs(V12[0] * V13[1] - V13[0]*V12[1]) / (d12 * d13);
    double sj2 = std::abs(V12[0] * V23[1] - V23[0]*V12[1]) / (d12 * d23);
    double sj3 = std::abs(V23[0] * V13[1] - V13[0]*V23[1]) / (d13 * d23);
    scaledJacobians[0] = std::min(sj1, sj2);
    scaledJacobians[0] = std::min(scaledJacobians[0], sj3);

    // Minimum angle
    double a1 = acos (V12.dot(V13) / (d12 * d13));
    double a2 = acos (-V12.dot(V23) / (d12 * d23));
    double a3 = acos (V23.dot(V13) / (d23 * d13));
    double aMin = std::min(a1, a2);
    minAngles[0] = std::min(aMin, a3);

    // eSkew
    double aMax = std::max(a1, a2);
    aMax = std::max(aMax, a3);
    double aIdeal = PI / 3;
    eSkews[0] = std::max((aMax - aIdeal / (PI - aIdeal)), (aIdeal - minAngles[0]) / aIdeal);

    // Area
    areas[0] = sqrt(s * (s-d12) * (s-d13) * (s-d23));

    // Aspect Ratio
    aspectRatios[0] = (d12 * d23 * d13) / (8 * (s - d12) * (s - d23) * (s - d13));

    // Skewness
    Vector2d midPoint1 = V2 + (V3 - V2) / 2;
    Vector2d midPoint2 = V3 + (V1 - V3) / 2;
    Vector2d midPoint3 = V1 + (V2 - V1) / 2;

    Vector2d lineNormal1 = midPoint1 - V1;
    Vector2d lineOrth1 = midPoint3 - midPoint2;
    double t1 = acos (lineNormal1.dot(lineOrth1) / (distance(V1, midPoint1) * distance(midPoint2, midPoint3)));
    double t2 = PI - t1;
    double tMin = std::min(t1, t2);

    Vector2d lineNormal2 = midPoint2 - V2;
    Vector2d lineOrth2 = midPoint1 - midPoint3;
    double t3 = acos (lineNormal2.dot(lineOrth2) / (distance(V2, midPoint2) * distance(midPoint1, midPoint3)));
    double t4 = std::min(t3, PI - t3);
    tMin = std::min(tMin, t4);

    Vector2d lineNormal3 = midPoint3 - V3;
    Vector2d lineOrth3 = midPoint2 - midPoint1;
    double t5 = acos (lineNormal3.dot(lineOrth3) / (distance(V3, midPoint3) * distance(midPoint1, midPoint2)));
    double t6 = std::min(t3, PI - t5);
    tMin = std::min(tMin, t6);

    skews[0] = PI/2 - tMin;
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