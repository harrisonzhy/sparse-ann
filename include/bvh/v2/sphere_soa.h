#ifndef BVH_V2_SPHERE_SOA_H
#define BVH_V2_SPHERE_SOA_H

#include "bvh/v2/vec.h"
#include "bvh/v2/ray.h"

#include <limits>
#include <utility>
#include <optional>

namespace bvh::v2 {

/// SphereSOA primitive defined by a (2D) center and squared radius.
template <typename T, size_t N=3>
struct SphereSoA {
    __m256 xs;
    __m256 ys;
    __m256 zs;
    __m256 rds;

    BVH_ALWAYS_INLINE SphereSoA() = default;

    // test intersection of 1 ray against 8 spheres (1x8)
    BVH_ALWAYS_INLINE int intersect_circle_1x8(const Ray<T, N>& ray) const {
        // broadcasted distance computation
        __m256 dist_x = _mm256_sub_ps(_mm256_set1_ps(ray.org[0]), this->xs);
        __m256 dist_y = _mm256_sub_ps(_mm256_set1_ps(ray.org[1]), this->ys);

        // squared distance from ray origin to sphere center
        __m256 dist_sq = _mm256_add_ps(
            _mm256_mul_ps(dist_x, dist_x),
            _mm256_mul_ps(dist_y, dist_y)
        );
        __m256 r_sq = _mm256_mul_ps(this->rds, this->rds);

        // compare squared distance to squared radius
        //  and return intersection indicators
        return _mm256_movemask_ps(_mm256_cmp_ps(dist_sq, r_sq, _CMP_LE_OQ));
    }

    BVH_ALWAYS_INLINE std::optional<std::tuple<int, __m256, __m256>> intersect_1x8(const Ray<T, N>& ray) const {
        __m256 ray_org[3] = {_mm256_set1_ps(ray.org[0]),
                             _mm256_set1_ps(ray.org[1]),
                             _mm256_set1_ps(ray.org[2])};
        __m256 ray_dir[3] = {_mm256_set1_ps(ray.dir[0]),
                             _mm256_set1_ps(ray.dir[1]),
                             _mm256_set1_ps(ray.dir[2])};

        __m256 ray_tmin = _mm256_set1_ps(ray.tmin);
        __m256 ray_tmax = _mm256_set1_ps(ray.tmax);

        __m256 oc_x = _mm256_sub_ps(ray_org[0], this->xs);
        __m256 oc_y = _mm256_sub_ps(ray_org[1], this->ys);
        __m256 oc_z = _mm256_sub_ps(ray_org[2], this->zs);

        // Compute b = 2 * dot(ray.dir, oc)
        __m256 b = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(ray_dir[0], oc_x),
                _mm256_mul_ps(ray_dir[1], oc_y)
            ),
            _mm256_mul_ps(ray_dir[2], oc_z)
        );
        b = _mm256_add_ps(b, b); // b *= 2

        // Compute c = dot(oc, oc) - radius^2
        __m256 c = _mm256_sub_ps(
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(oc_x, oc_x),
                    _mm256_mul_ps(oc_y, oc_y)
                ),
                _mm256_mul_ps(oc_z, oc_z)
            ),
            _mm256_mul_ps(this->rds, this->rds)
        );

        // Compute delta = b^2 - 4 * a * c (a = 1 for normalized rays)
        __m256 delta = _mm256_sub_ps(
            _mm256_mul_ps(b, b),
            _mm256_mul_ps(_mm256_set1_ps(4.f), c)
        );

        // Mask for delta >= 0
        __m256 mask_delta = _mm256_cmp_ps(delta, _mm256_setzero_ps(), _CMP_GE_OQ);

        // Compute sqrt(delta)
        __m256 sqrt_delta = _mm256_sqrt_ps(delta);

        // Compute t0 and t1
        __m256 inv_a = _mm256_set1_ps(-0.5f); // -0.5 / a, where a = 1
        __m256 t0 = _mm256_mul_ps(_mm256_add_ps(b, sqrt_delta), inv_a);
        __m256 t1 = _mm256_mul_ps(_mm256_sub_ps(b, sqrt_delta), inv_a);
        t0 = _mm256_max_ps(t0, ray_tmin);
        t1 = _mm256_min_ps(t1, ray_tmax);

        // Mask for t0 <= t1
        __m256 mask_tx = _mm256_cmp_ps(t0, t1, _CMP_LT_OQ);

        // Final mask: delta >= 0 and t0 <= t1
        int mask_valid = _mm256_movemask_ps(_mm256_and_ps(mask_delta, mask_tx));

        // If no intersections are valid, return std::nullopt
        if (mask_valid == 0) {
            return std::nullopt;
        }

        // Return valid t0 and t1
        return std::make_optional(std::make_tuple(mask_valid, t0, t1));
    }
};

} // namespace bvh::v2

#endif
