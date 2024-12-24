#include <immintrin.h>
#include <cmath>

namespace tracer {

struct SphereSoA {
    __m256 xs;
    __m256 ys;
    __m256 r_sqs;
};

struct Ray {
    Ray() : x(0), y(0) {}
    Ray(float x_, float y_) : x(x_), y(y_) {}
    float x;
    float y;
};

// test intersection of a single ray against 8 spheres (1x8)
int intersect_1x8(const Ray& ray, const SphereSoA& spheres) {
    // broadcast ray
    __m256 ray_org_x = _mm256_set1_ps(ray.x);
    __m256 ray_org_y = _mm256_set1_ps(ray.y);

    __m256 dist_x = _mm256_sub_ps(ray_org_x, spheres.xs);
    __m256 dist_y = _mm256_sub_ps(ray_org_y, spheres.ys);

    // squared distance from the ray origin to the sphere center
    __m256 dist_sq = _mm256_add_ps(
        _mm256_mul_ps(dist_x, dist_x),
        _mm256_mul_ps(dist_y, dist_y)
    );

    // compare squared distance to squared radius
    //  and return intersection indicators
    return _mm256_movemask_ps(_mm256_cmp_ps(dist_sq, spheres.r_sqs, _CMP_LE_OQ));
}

}; // namespace tracer
