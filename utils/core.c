#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define V_DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
#define V_CROSS(v1, v2) v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]
#define V_LEN(v) sqrt(V_DOT(v, v))

#define V_ADD(v1, v2) v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]
#define V_SUB(v1, v2) v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]
#define V_MADD(v1, v2, s) v1[0] + v2[0]*s, v1[1] + v2[1]*s, v1[2] + v2[2]*s

#define T_CORRECTION 0.00001
#define RAY_ARGS double* rayOrigin, double* rayDirection, double rayT

const double None = 0.0;

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


float triangleIntersection(RAY_ARGS, double* position, double* normal, double* A, double* B, double* C, double area) {
    double dn = V_DOT(rayDirection, normal);
    if (dn == 0.) return None;

    double po[] = {V_SUB(position, rayOrigin)};
    double t = V_DOT(po, normal) / dn;
    if (t < 0. || rayT < t) return None;

    double p[] = {V_MADD(rayOrigin, rayDirection, t)};
    double AP[] = {V_SUB(A, p)};
    double BP[] = {V_SUB(B, p)};
    double CP[] = {V_SUB(C, p)};

    double AB[] = {V_CROSS(AP, BP)};
    double CA[] = {V_CROSS(CP, AP)};

    double c1 = V_DOT(AB, normal) / area;
    double c2 = V_DOT(CA, normal) / area;
    double c3 = 1 - c1 - c2;

    if (c1 < -T_CORRECTION || c2 < -T_CORRECTION || c3 < -T_CORRECTION) return None;

    // double a1[] = {V_CROSS(BP, CP)};
    // double a2[] = {V_CROSS(CP, AP)};
    // double a3[] = {V_CROSS(BP, AP)};

    // if (abs(V_LEN(a1) + V_LEN(a2) + V_LEN(a3) - area) > 0.1) {
    //     return None;
    // }
    // printf("%.2f\n", abs(V_LEN(a1) + V_LEN(a2) + V_LEN(a3) - area));

    return t - T_CORRECTION;
}
