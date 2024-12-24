#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <limits>

// #include "sparse_rt.hpp"

// using Point2D = Eigen::Vector2d;
// // using Transform2D = Eigen::Matrix3d;

// Transform2D calculate_transform(const std::vector<Point2D>& src, const std::vector<Point2D>& dst) {
//     // Compute centroids
//     Point2D centroid_src = Point2D::Zero();
//     Point2D centroid_dst = Point2D::Zero();
//     for (auto i = 0; i < src.size(); ++i) {
//         centroid_src += src[i];
//         centroid_dst += dst[i];
//     }
//     centroid_src /= src.size();
//     centroid_dst /= dst.size();

//     // Center the points
//     Eigen::MatrixXd centered_src(2, src.size());
//     Eigen::MatrixXd centered_dst(2, dst.size());
//     for (size_t i = 0; i < src.size(); ++i) {
//         centered_src.col(i) = src[i] - centroid_src;
//         centered_dst.col(i) = dst[i] - centroid_dst;
//     }

//     // Compute the SVD
//     Eigen::Matrix2d covariance = centered_dst * centered_src.transpose();
//     Eigen::JacobiSVD<Eigen::Matrix2d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
//     Eigen::Matrix2d rotation = svd.matrixU() * svd.matrixV().transpose();

//     // Ensure a proper rotation (det = 1)
//     if (rotation.determinant() < 0) {
//         Eigen::Matrix2d correction = Eigen::Matrix2d::Identity();
//         correction(1, 1) = -1;
//         rotation = svd.matrixU() * correction * svd.matrixV().transpose();
//     }

//     // Compute the translation
//     Point2D translation = centroid_dst - rotation * centroid_src;

//     // Build the transformation matrix
//     Transform2D transform = Transform2D::Identity();
//     transform.block<2, 2>(0, 0) = rotation;
//     transform.block<2, 1>(0, 2) = translation;

//     std::cout << transform << std::endl;

//     return transform;
// }


// // RANSAC algorithm to find the best transformation between subspace `this_m` and `next_m`
// template<typename T>
// Transform2D CoreRT::ransac_align(const T* data, int this_m, int next_m, int max_iter, float threshold) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, n - 1);

//     // separate into points per subspace
//     std::vector<Point2D> src;
//     std::vector<Point2D> dst;

//     for (auto i = 0; i < n; ++i) {
//         const auto data_point = data + i * dim;
//         src.emplace_back(data_point[this_m * sub_dim], data_point[this_m * sub_dim + 1]); // x coord, y coord
//         dst.emplace_back(data_point[next_m * sub_dim], data_point[next_m * sub_dim + 1]); // x coord, y coord
//     }

//     Transform2D best_transform = Transform2D::Identity();
//     auto best_inliers = 0;

//     for (auto iter = 0; iter < iterations; ++iter) {
//         // Randomly select two points
//         int idx1 = dis(gen);
//         int idx2 = dis(gen);
//         while (idx1 == idx2) {
//             idx2 = dis(gen);
//         }

//         const std::vector<Point2D> src_sample = {src[idx1], src[idx2]};
//         const std::vector<Point2D> dst_sample = {dst[idx1], dst[idx2]};

//         // compute transformation matrix for these points
//         Transform2D transform = calculate_transform(src_sample, dst_sample);

//         // count inliers
//         auto inliers = 0;
//         for (auto i = 0; i < n; ++i) {
//             Eigen::Vector3d src_h(src[i].x(), src[i].y(), 1.0);
//             Eigen::Vector3d transformed = transform * src_h;
//             const Point2D projected(transformed.x(), transformed.y());
//             const float distance = (projected - dst[i]).norm();
//             if (distance < threshold) {
//                 ++inliers;
//             }
//         }

//         // update best model
//         if (inliers > best_inliers) {
//             best_inliers = inliers;
//             best_transform = transform;
//         }
//     }

//     std::cout << "best_inliers: " << best_inliers << std::endl;
//     return best_transform;
// }
