#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define V_DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])

#define V_ADD(v1, v2) v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]
#define V_SUB(v1, v2) v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]
#define V_MADD(v1, v2, s) v1[0] + v2[0]*s, v1[1] + v2[1]*s, v1[2] + v2[2]*s

#define T_CORRECTION 0.00001
#define RAY_ARGS double* rayOrigin, double* rayDirection, double rayT

const double None = -1.0;

float sphereIntersection(RAY_ARGS, double* position, double radius) {
    double co[] = {V_SUB(rayOrigin, position)};

    double b = 2.*V_DOT(co, rayDirection);
    double c = V_DOT(co, co) - (radius*radius);
    double delta = (b*b) - 4.*c;
    if (delta < 0.) return None;

    delta = sqrt(delta);

    double t1 = (-b + delta) / 2;
    double t2 = (-b - delta) / 2;
    double t = rayT;
    if (0 < t1 && t1 < t) t = t1;
    if (0 < t2 && t2 < t) t = t2;
    if (t == rayT) return None;

    return t - T_CORRECTION;
}
