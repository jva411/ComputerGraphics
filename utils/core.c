#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define V_DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
#define V_CROSS(v1, v2) {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]}
#define V_LEN(v) sqrt(V_DOT(v, v))

#define V_ADD(v1, v2) {v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]}
#define V_SUB(v1, v2) {v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]}
#define V_MADD(v1, v2, s) {v1[0] + v2[0]*s, v1[1] + v2[1]*s, v1[2] + v2[2]*s}

#define T_CORRECTION 0.00001
#define RAY_ARGS double* rayOrigin, double* rayDirection, double rayT

const double None = 0.0;


float sphereIntersection(RAY_ARGS, double* position, double radius) {
    double co[] = V_SUB(rayOrigin, position);

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

    double po[] = V_SUB(position, rayOrigin);
    double t = V_DOT(po, normal) / dn;
    if (t < 0. || rayT < t) return None;

    double p[] = V_MADD(rayOrigin, rayDirection, t);
    double AP[] = V_SUB(A, p);
    double BP[] = V_SUB(B, p);
    double CP[] = V_SUB(C, p);

    double AB[] = V_CROSS(AP, BP);
    double CA[] = V_CROSS(CP, AP);

    double c1 = V_DOT(AB, normal) / area;
    double c2 = V_DOT(CA, normal) / area;
    double c3 = 1 - c1 - c2;

    if (c1 < -T_CORRECTION || c2 < -T_CORRECTION || c3 < -T_CORRECTION) return None;

    return t - T_CORRECTION;
}


float cylinderIntersection(RAY_ARGS, double* position, double* axis, double radius, double height) {
    double po[] = V_SUB(rayOrigin, position);
    double v[] = V_MADD(po, axis, -V_DOT(po, axis));
    double w[] = V_MADD(rayDirection, axis, -V_DOT(rayDirection, axis));

    double a = V_DOT(w, w);
    if (a==0.) return None;

    double b = V_DOT(v, w);
    double c = V_DOT(v, v) - radius*radius;
    double delta = b*b - a*c;
    if (delta < 0.) return None;

    delta = sqrt(delta);
    double t1 = (-b - delta) / a;
    double t2 = (-b + delta) / a;
    double p1[] = V_MADD(rayOrigin, rayDirection, t1);
    double p2[] = V_MADD(rayOrigin, rayDirection, t2);
    double po1[] = V_SUB(position, p1);
    double po2[] = V_SUB(position, p2);
    double dp1 = V_DOT(po1, axis);
    double dp2 = V_DOT(po2, axis);

    double t = rayT;
    if (0 < t1 && t1 < t && 0 <= dp1 && dp1 <= height) t = t1;
    if (0 < t2 && t2 < t && 0 <= dp2 && dp2 <= height) t = t2;
    if (t == rayT) return None;

    return t - T_CORRECTION;
}


float coneIntersection(RAY_ARGS, double* position, double* axis, double cos2, double height) {
    double v[] = V_SUB(position, rayOrigin);
    double dn = V_DOT(rayDirection, axis);
    double vn = V_DOT(v, axis);

    double a = (dn*dn) - (V_DOT(rayDirection, rayDirection) * cos2);
    if (a==0.) return None;

    double b = (V_DOT(v, rayDirection) * cos2) - (vn * dn);
    double c = (vn*vn) - (V_DOT(v, v) * cos2);
    double delta = b*b - a*c;

    if (delta < 0.) return None;

    delta = sqrt(delta);
    double t1 = (-b - delta) / a;
    double t2 = (-b + delta) / a;
    double p1[] = V_MADD(rayOrigin, rayDirection, t1);
    double p2[] = V_MADD(rayOrigin, rayDirection, t2);
    double po1[] = V_SUB(position, p1);
    double po2[] = V_SUB(position, p2);
    double dp1 = V_DOT(po1, axis);
    double dp2 = V_DOT(po2, axis);

    double t = rayT;
    if (0 < t1 && t1 < t && 0 <= dp1 && dp1 <= height) t = t1;
    if (0 < t2 && t2 < t && 0 <= dp2 && dp2 <= height) t = t2;
    if (t == rayT) return None;

    return t - T_CORRECTION;
}

float circleIntersection(RAY_ARGS, double* position, double* normal, double radius) {
    double dn = V_DOT(rayDirection, normal);
    if (dn == 0.) return None;

    double po[] = V_SUB(position, rayOrigin);
    double t = V_DOT(po, normal) / dn;
    if (t < 0. || rayT < t) return None;

    double p[] = V_MADD(rayOrigin, rayDirection, t);
    double pp[] = V_SUB(p, position);
    if (V_LEN(pp) > radius) return None;

    return t - T_CORRECTION;
}
