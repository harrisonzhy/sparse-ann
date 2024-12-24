#include <cstring>
#include <algorithm>
#include <queue>
#include <execution>
#include <set>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexPQ.h>
#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/VectorTransform.h>
#include <faiss/utils/distances.h>

#include <Eigen/Dense>

#include "utils.hpp"
#include "distance.hpp"
// #include "ray_tracer.hpp"

using namespace std;

using Transform = Eigen::Matrix<float, 3, 3>;

using Scalar = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Sph     = bvh::v2::Sphere<Scalar, 3>;
using SphSoA  = bvh::v2::SphereSoA<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

Transform compute_homography(const std::vector<Eigen::Vector3f>& src, const std::vector<Eigen::Vector3f>& dst);

template <typename T=float, typename CT=uint8_t>
class CoreRT {
private:
  int nclusters; // number of clusters in each sub-space
  int dim; // number of dimensions of each vector
  int m; // number of sub-spaces
  int sub_dim; // number of dimensions of each subvector
  size_t n; // number of vectors in the database

  int nlist;
  float radius;
  int nspheres;

  faiss::IndexFlatL2* quantizer; // trained quantizer (and sub-quantizers)
  faiss::IndexPQ* index; // trained index
  vector<vector<CT>> codebook;
  vector<T*> centroids; // m * ncluster * sub_dim; PQ centroids
  vector<vector<T>> lookup_table; // m * ncluster
  vector<std::pair<T*, const faiss::idx_t*>> coarse_centroids;
  vector<int> coarse_centroids_size;

  Bvh* rt_bvh;
  vector<SphSoA> rt_spheres_soa;
  vector<Sph> rt_spheres;
  vector<vector<int>> rt_cluster_mapping;  // points within a sphere
  vector<Transform> rt_subspace_transforms;

public:
  CoreRT(int nclusters_, int dim_, size_t n_, const T* data_vectors, int nlist_, float radius_) 
    : nclusters(nclusters_),
      dim(dim_), 
      m(dim_ / 2),
      n(n_),
      nlist(nlist_),
      radius(radius_),
      quantizer(nullptr), 
      index(nullptr),
      nspheres(m * nclusters) {
    
    sub_dim = dim / m;

    assert(dim % m == 0);
    assert(nlist > 0);
    assert(nclusters <= 256);

    printf("CoreRT: m=%d, nclusters=%d, dim=%d, radius=%f, n=%lu\n", m, nclusters, dim, radius, n);

    centroids.resize(m, 0);
    coarse_centroids.resize(nlist);
    coarse_centroids_size.resize(nlist);
    codebook.resize(n, std::vector<CT>(m, 0));
    lookup_table.resize(m, std::vector<T>(nclusters, 0));
    rt_cluster_mapping.resize(nspheres);
    rt_subspace_transforms.resize(m, Transform::Identity());

    train_centroids(data_vectors);
    find_subspace_transforms(data_vectors);
    construct_BVHPQ(data_vectors);
  }

  
  void train_centroids(const T* data) {
    Timer t;
    
    printf("Train centroids ... \n");
    t.Start();
    // quantizer = new faiss::IndexFlatL2(dim);
    // index = new faiss::IndexIVFPQ(quantizer, dim, nlist, m, 63 - __builtin_clzll(nclusters));
    index = new faiss::IndexPQ(dim, m, 63 - __builtin_clzll(nclusters));
    index->train(n, data);
    index->add(n, data);

    // centroid table (m, nclusters, sub_dim)
    for (auto i = 0; i < m; ++i) {
      centroids[i] = index->pq.get_centroids(i, 0);
    }

    t.Stop();
    printf("time = %f\n", t.Seconds());
  }

  void search_one_shot(int nq_, const T* queries, int k, int nprobe, int* results, std::vector<bool>& visited) {
    assert(nprobe == 1);
    constexpr auto nq = 1;

    vector<bvh::v2::SmallStack<Bvh::Index, bvh_stack_size>> stacks(nq);
    vector<T> dis_tables(nq * m * nclusters);

    std::vector<T> D(nq * m * nclusters);
    faiss::HeapArray<faiss::CMax<T, int>> res = { nq, k, results, D.data() };

    #pragma omp parallel for if (nq > 1)
    for (auto qid = 0; qid < nq; ++qid) {
      const auto query = queries + qid * dim;
      T* dis_table = dis_tables.data() + qid * m * nclusters;

      // compute lookup tables
      for (auto j = 0; j < m; ++j) {
        faiss::fvec_L2sqr_ny(
          dis_table + j * nclusters,
          query + j * sub_dim,
          centroids[j],
          sub_dim,
          nclusters);
      }

      T*   __restrict heap_dis = res.val + qid * k;
      int* __restrict heap_ids = res.ids + qid * k;
      faiss::heap_heapify<faiss::CMax<T, int>>(k, heap_dis, heap_ids);

      Timer t;
      
      // std::vector<int> candidates;
      // std::unordered_set<int> candidates;
      
      // traverse the scene
      // t.Start();
        
      vector<bool> batch_intersected(rt_spheres_soa.size(), false);
      // T st0[tile_width] __attribute__((aligned(32)));
      // T st1[tile_width] __attribute__((aligned(32)));

      // #pragma omp parallel for
      for (auto j = 0; j < m; ++j) {
        const auto x = query[j * sub_dim];
        const auto y = query[j * sub_dim + 1];
        const auto ray = Ray(Vec3(x, y, 0),
                              Vec3(0, 0, 1));

        // traverse the BVH
        auto prim_id = bvh_invalid_id;
        rt_bvh->intersect<true, /* isAnyHit */
                          false /* isRobust */
          >(ray, rt_bvh->get_root().index, stacks[qid], [&](size_t begin, size_t end) {

          for (auto s_ = begin; s_ < end; ++s_) {

            const auto s = rt_bvh->prim_ids[s_] / tile_width;
            if (!batch_intersected[s]) {
              batch_intersected[s] = true;
              auto ret_val = rt_spheres_soa[s].intersect_1x8(ray);
              if (ret_val) {
                prim_id = s;
                auto [intersect_mask, t0, t1] = *ret_val;
                if (!intersect_mask) {
                  continue;
                }

                // distance accumulation
                for (auto w = 0; w < tile_width; ++w) {
                  if (intersect_mask & (1 << w)) {
                    const auto& neighborhood = rt_cluster_mapping[s * tile_width + w];
                    for (auto p = 0; p < neighborhood.size(); ++p) {
                      auto point = neighborhood[p];
                      if (visited[point]) {
                        continue;
                      }
                      visited[point] = true;
                      auto codes = index->codes.data() + point * m;
                      const T* dt = dis_table;
                      T dis = 0;
                      for (auto j = 0; j < m; j += 4) {
                        T dism = 0;
                        dism = dt[*codes++];
                        dt += nclusters;
                        dism += dt[*codes++];
                        dt += nclusters;
                        dism += dt[*codes++];
                        dt += nclusters;
                        dism += dt[*codes++];
                        dt += nclusters;
                        dis += dism;
                      }
                      if (faiss::CMax<T, int>::cmp(heap_dis[0], dis)) {
                        faiss::heap_replace_top<faiss::CMax<T, int>>(k, heap_dis, heap_ids, dis, point);
                      }
                    }
                  }
                }
              }
            }
            
            // // this doesn't scale
            // const auto sph_idx = rt_bvh->prim_ids[s];
            // if (auto ret_val = rt_spheres[sph_idx].intersect(ray)) {
            //   auto [t0, t1] = *ret_val;
            //   const auto& neighborhood = rt_cluster_mapping[sph_idx];
            //   // std::cout << t0 << " " << t1 << std::endl;
            //   for (auto p = 0; p < neighborhood.size(); ++p) {
            //     candidates.insert(neighborhood[p]);
            //   }

            // std::set_union(
            //     std::begin(candidates), std::end(candidates),
            //     std::begin(neighborhood), std::end(neighborhood),
            //     std::back_inserter(tmp_));
            // candidates = std::move(tmp_);

            // prim_id = sph_idx;
          }
          return prim_id != bvh_invalid_id;
        });
      }

      // t.Stop();

      // cout << "bvh time: " << t.Seconds() << endl;

      // t.Start();

      // std::cout << candidates.size() << std::endl;
      // std::vector<int> tmp_(candidates.size());
      // tmp_.assign(begin(candidates), end(candidates));
      // for (auto c : candidates) {
      //   tmp_.push_back(c);
      // }
      // sort(begin(tmp_), end(tmp_));

      // #pragma omp parallel for
      // for (auto p = 0; p < tmp_.size(); ++p) {
      //   auto point = tmp_[p];
      //   auto codes = index->codes.data() + point * m;
      //   const T* dt = dis_table;
      //   T dis = 0;
      //   for (auto j = 0; j < m; j += 4) {
      //     T dism = 0;
      //     dism = dt[*codes++];
      //     dt += nclusters;
      //     dism += dt[*codes++];
      //     dt += nclusters;
      //     dism += dt[*codes++];
      //     dt += nclusters;
      //     dism += dt[*codes++];
      //     dt += nclusters;
      //     dis += dism;
      //   }
      //   if (faiss::CMax<T, int>::cmp(heap_dis[0], dis)) {
      //     faiss::heap_replace_top<faiss::CMax<T, int>>(k, heap_dis, heap_ids, dis, point);
      //   }
      // }

      // faiss::heap_reorder<faiss::CMax<T, int>>(k, heap_dis, heap_ids);

      // t.Stop();
      // cout << "dist accum time: " << t.Seconds() << endl;
    }
  }


  void faiss_search(int nq, const T* queries, int k, int nprobe, int* results) {
    std::vector<float> D(nq * m * nclusters);
    assert(nprobe == 1);

    vector<T> dis_tables(nq * m * nclusters);
    faiss::HeapArray<faiss::CMax<T, int>> res = { nq, k, results, D.data() };
    // index->pq.compute_distance_tables(nq, queries, dis_tables.data());
    
    #pragma omp parallel for if (nq > 1)
    for (auto qid = 0; qid < nq; ++qid) {
      const auto query = queries + qid * dim;
      T* dis_table = dis_tables.data() + qid * m * nclusters;

      Timer t;

      for (auto j = 0; j < m; j++) {
        faiss::fvec_L2sqr_ny(
                dis_table + j * nclusters,
                query + j * sub_dim,
                centroids[j],
                sub_dim,
                nclusters);
      }

      // compute distance tables (LUT)
      auto codes = index->codes.data();
      int* __restrict heap_ids = res.ids + qid * k;
      T*   __restrict heap_dis = res.val + qid * k;

      faiss::heap_heapify<faiss::CMax<T, int>>(k, heap_dis, heap_ids);

      t.Start();
      for (auto i = 0; i < n; ++i) {
        const T* dt = dis_table;
        T dis = 0;
        for (auto j = 0; j < m; j += 4) {
            T dism = 0;
            dism = dt[*codes++];
            dt += nclusters;
            dism += dt[*codes++];
            dt += nclusters;
            dism += dt[*codes++];
            dt += nclusters;
            dism += dt[*codes++];
            dt += nclusters;
            dis += dism;
        }
        if (faiss::CMax<T, int>::cmp(heap_dis[0], dis)) {
          faiss::heap_replace_top<faiss::CMax<T, int>>(k, heap_dis, heap_ids, dis, i);
        }
      }
      t.Stop();
      // std::cout << "dist accum time: " << t.Seconds() << std::endl;

      faiss::heap_reorder<faiss::CMax<T, int>>(k, heap_dis, heap_ids);
    }
  }

  void find_subspace_transforms(const T* data) {
    Timer t;
    printf("Finding subspace transforms...\n");
    t.Start();
    
    // list of (m-1) homographies that go from I_{m_} to I_{0}
    #pragma omp parallel for
    for (auto j = 1; j < m; ++j) {
      // rt_subspace_transforms[j] = trimmed_icp(data, j, 0);
    }
    t.Stop();
    printf("time = %f\n", t.Seconds());
  }


  Transform trimmed_icp(const T* data, int src_m, int dst_m, int max_iter = 1000, float trim_ratio = 0.8) {
      std::vector<Eigen::Vector3f> src(nclusters);
      std::vector<Eigen::Vector3f> dst(nclusters);
      std::vector<int> residual_weights_src(nclusters);

      constexpr auto min_correspondences = 8;

      // prepare cluster centroid correspondences between `src_m` and `dst_m`
      for (auto i = 0; i < nclusters; ++i) {
        src[i] = Eigen::Vector3f(centroids[src_m][i * sub_dim], centroids[src_m][i * sub_dim + 1], 1); // x, y, 1
        dst[i] = Eigen::Vector3f(centroids[dst_m][i * sub_dim], centroids[dst_m][i * sub_dim + 1], 1); // x, y, 1
        residual_weights_src[i] = max(size_t{1}, rt_cluster_mapping[src_m * nclusters + i].size());
      }

      Transform best_transform = Transform::Identity();
      float last_error = std::numeric_limits<float>::max();

      for (auto iter = 0; iter < max_iter; ++iter) {
        // 1. find closest points (correspondences)
        std::vector<std::pair<int, float>> residuals; // (index, residual)
        for (auto i = 0; i < nclusters; ++i) {
          Eigen::Vector3f transformed_point = best_transform * src[i];
          float min_residual = std::numeric_limits<float>::max();

          for (auto j = 0; j < nclusters; ++j) {
            min_residual = min((dst[j] - transformed_point).squaredNorm(), min_residual);
          }
          residuals.emplace_back(i, min_residual);
        }

        // 2. sort residuals based on weight and trim outliers
        const auto stabilizer = *std::max_element(begin(residual_weights_src), end(residual_weights_src));
        std::sort(residuals.begin(), residuals.end(), 
          [&](const auto& a, const auto& b) {
              return a.second * (stabilizer / residual_weights_src[a.first])
                   < b.second * (stabilizer / residual_weights_src[b.first]);
          });

        const auto num_trimmed = max(static_cast<int>(nclusters * trim_ratio), min_correspondences);
        std::vector<Eigen::Vector3f> trimmed_src;
        std::vector<Eigen::Vector3f> trimmed_dst;

        for (auto i = 0; i < num_trimmed; ++i) {
          const auto idx = residuals[i].first;
          trimmed_src.push_back(src[idx]);
          trimmed_dst.push_back(dst[idx]);
        }

        // 3. compute new transformation using trimmed points
        Transform H = compute_homography(trimmed_src, trimmed_dst);
        if (H.determinant() == 0) {
          H = Transform::Identity();
        }

        // 4. evaluate error with trimmed correspondences
        float total_error = 0;
        for (int i = 0; i < num_trimmed; ++i) {
          Eigen::Vector3f transformed_point = H * trimmed_src[i];
          total_error += (trimmed_dst[i] - transformed_point).squaredNorm();
        }
        total_error /= num_trimmed;

        // 5. check for convergence
        if (std::abs(last_error - total_error) < 1e-6) {
          break;
        }

        last_error = total_error;
        best_transform = H;
      }

    return best_transform;
  }


  // RANSAC algorithm to find the best transformation between subspaces `src_m` and `dst_m`
  Transform ransac(const T* data, int src_m, int dst_m, int n_iter = 25000, float eps = 1.0) {
    constexpr auto num_correspondences = 8;
    
    std::vector<Eigen::Vector3f> src(nclusters);
    std::vector<Eigen::Vector3f> dst(nclusters);

    // prepare cluster centroid correspondences between `src_m` and `dst_m` subspaces
    for (auto i = 0; i < nclusters; ++i) {
      src[i] = Eigen::Vector3f(centroids[src_m][i * sub_dim], centroids[src_m][i * sub_dim + 1], 1); // x coord, y coord 
      dst[i] = Eigen::Vector3f(centroids[dst_m][i * sub_dim], centroids[dst_m][i * sub_dim + 1], 1); // x coord, y coord
    }

    // weighted shuffle based on cluster density in each subspace
    auto weighted_random_shuffle = \
      [](auto& elements, const auto& weights) {
        std::vector<std::pair<double, typename std::decay<decltype(elements[0])>::type>> scores;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0, 1);

        const auto stabilizer = *std::max_element(begin(weights), end(weights));
        for (auto i = 0; i < elements.size(); ++i) {
            const double score = (-std::log(dist(gen)) * stabilizer) / weights[i]; // exponential sampling
            scores.emplace_back(score, elements[i]);
        }

        // sort based on the random scores (higher weight means more likely to be in front)
        std::sort(begin(scores), end(scores),
          [](const auto& a, const auto& b) {
            return a.first < b.first;
          });
        for (auto i = 0; i < elements.size(); ++i) {
          elements[i] = scores[i].second;
        }
      };

    std::vector<int> shuffle_weights_src(nclusters);
    std::vector<int> shuffle_weights_dst(nclusters);
    for (auto i = 0; i < nclusters; ++i) {
      shuffle_weights_src[i] = rt_cluster_mapping[src_m * nclusters + i].size();
      shuffle_weights_dst[i] = rt_cluster_mapping[dst_m * nclusters + i].size();
    }

    Transform best_model = Transform::Identity();
    auto best_inliers = 0;
    for (auto iter = 0; iter < n_iter; ++iter) {
      weighted_random_shuffle(src, shuffle_weights_src);
      weighted_random_shuffle(dst, shuffle_weights_dst);

      const std::vector<Eigen::Vector3f> src_corres{std::begin(src), std::begin(src) + num_correspondences};
      const std::vector<Eigen::Vector3f> dst_corres{std::begin(dst), std::begin(dst) + num_correspondences};

      auto H = compute_homography(src_corres, dst_corres);
      if (H.determinant() == 0) {
        H = Transform::Identity();
      }

      // count inliers
      auto inliers = 0;
      for (auto i = 0; i < nclusters; ++i) {
        const auto p1 = H * src[i];
        const auto p2 = dst[i];
        inliers += ((p2 - p1).squaredNorm() < eps);
      }

      if (inliers > best_inliers) {
        best_model = H;
        best_inliers = inliers;
      }
    }

    return best_model;
  }


  void construct_BVHPQ(const T* data) {
    Timer t;

    t.Start();
    printf("Build BVH ...\n");

    // construct `nclusters` spheres in each subspace
    for (auto i = 0; i < m; ++i) {
      // z_0 = 1 * radius + 1
      // z_1 = 4 * radius + 1
      // z_2 = 7 * radius + 1
      // ...
      const auto& sub_transform = rt_subspace_transforms[i];
      const auto c_i = centroids[i];
      float z = (3 * i + 1) * radius + 1; // z coord for spheres

      for (auto j = 0; j < nclusters; j += tile_width) {
        T scx[tile_width] __attribute__((aligned(32)));
        T scy[tile_width] __attribute__((aligned(32)));
        T srd[tile_width] __attribute__((aligned(32)));
        
        for (auto k = 0; k < tile_width; ++k) {
          auto center = c_i + (j + k) * sub_dim;
          // create subspace plane-aligned scene
          Eigen::Vector3f txf_cluster = sub_transform * Eigen::Vector3f(center[0], center[1], 0);
          float x = txf_cluster[0];
          float y = txf_cluster[1];
          scx[k] = x;
          scy[k] = y;
          srd[k] = radius;
        }

        SphSoA sphere_soa;
        sphere_soa.xs  = _mm256_set_ps(scx[0], scx[1], scx[2], scx[3], scx[4], scx[5], scx[6], scx[7]);
        sphere_soa.ys  = _mm256_set_ps(scy[0], scy[1], scy[2], scy[3], scy[4], scy[5], scy[6], scy[7]);
        sphere_soa.rds = _mm256_set_ps(srd[0], srd[1], srd[2], srd[3], srd[4], srd[5], srd[6], srd[7]);
        rt_spheres_soa.push_back(sphere_soa);

        for (auto k = 0; k < tile_width; ++k) {
          rt_spheres.emplace_back(Vec3(scx[k], scy[k], 0), srd[k]);
        }
      }
    }

    // construct BVH tree on the scene in parallel
    //  each node is a SphereSoA
    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    const auto nspheres_soa = rt_spheres_soa.size();
    // vector<BBox> rt_bboxes(nspheres_soa);
    // vector<Vec3> rt_centers(nspheres_soa);
    vector<BBox> rt_bboxes(nspheres);
    vector<Vec3> rt_centers(nspheres);
    executor.for_each(0, nspheres,
      [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
          // const auto& sphere_soa = rt_spheres_soa[i];
          // T scx[tile_width] __attribute__((aligned(32))); _mm256_store_ps(scx, sphere_soa.xs);
          // T scy[tile_width] __attribute__((aligned(32))); _mm256_store_ps(scy, sphere_soa.ys);
          // T srd[tile_width] __attribute__((aligned(32))); _mm256_store_ps(srd, sphere_soa.rds);
          
          // const int m_ = (i * tile_width) / nclusters;
          // float z = (3 * m_ + 1) * radius + 1; // z coord for sphere

          // const auto init_sphere = Sph{ Vec3(scx[0], scy[0], z), srd[0] };
          // auto bbox_soa = init_sphere.get_bbox();
          // auto center_soa = init_sphere.get_center();

          // for (auto j = 1; j < tile_width; ++j) {
          //   Sph s{ Vec3(scx[j], scy[j], z), srd[j] };
          //   bbox_soa.extend(s.get_bbox());
          //   center_soa = center_soa + s.get_center();
          // }
          // center_soa = center_soa / (T)tile_width;
          // rt_bboxes[i] = bbox_soa;
          // rt_centers[i] = center_soa;

          rt_bboxes[i] = rt_spheres[i].get_bbox();
          rt_centers[i] = rt_spheres[i].get_center();
        }
    });

    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
    rt_bvh = new Bvh(bvh::v2::DefaultBuilder<Node>::build(thread_pool, rt_bboxes, rt_centers, config));

    t.Stop();
    double runtime = t.Seconds();
    printf("time = %f\n", runtime);
    
    t.Start();
    printf("Build codebook ...\n");

    // faster codebook construction and sphere assignment
    // (1) build codebook
    // (2) assign subvectors to spheres in each subspace by vec_id
    vector<vector<bool>> assignments(n, vector<bool>(nspheres, false));
    const auto radius_sq = radius * radius;

    #pragma omp parallel for
    for (auto i = 0; i < n; ++i) { // for each vector `x_i`
      auto x_i = data + i * dim;
      for (auto j = 0; j < m; ++j) { // for each subspace `j`
        float min_dist = __builtin_inff();
        uint16_t best_idx = 0;
        auto sub_xi = x_i + j * sub_dim;
        auto c_j = centroids[j];
        for (auto k = 0; k < nclusters; ++k) { // for each cluster `k`
          auto c_jk = c_j + k * sub_dim;
          auto dist = compute_distance_squared<T>(sub_dim, sub_xi, c_jk);
          if (dist < radius_sq) {
            auto s = j * nclusters + k;
            assignments[i][s] = true;
          }
          if (dist < min_dist) {
            min_dist = dist;
            best_idx = k;
          }
        }
        codebook[i][j] = best_idx;
      }
    }

    #pragma omp parallel for
    for (auto i = 0; i < n; ++i) {
      for (auto j = 0; j < m; ++j) {
        for (auto k = 0; k < nclusters; ++k) {
          auto s_id = j * nclusters + k;
          if (assignments[i][s_id]) {
            #pragma omp critical
            {
              rt_cluster_mapping[s_id].push_back(i);
            }
            // unassign this point from other clusters in this subspace `j`
            for (auto k_ = 0; k_ < nclusters; ++k_) {
              assignments[i][j * nclusters + k_] = false;
            }
          }
        }
      }
    }

    t.Stop();
    runtime = t.Seconds();
    printf("time = %f\n", runtime); 

    printf("Coalesce spheres...\n");
    t.Start();

    // create a bidirectional graph on the spheres in each subspace
    //  edge weight is distance
    vector<vector<vector<std::tuple<Sph*, float, int>>>> \
      sphere_graph(m, vector<vector<std::tuple<Sph*, float, int>>>(nspheres)); // <sphere, distance to neighbor, id>
    vector<std::priority_queue<std::tuple<int, int, Sph*>>> \
      sphere_nbr_rank(m); // <rank, cluster, sphere>

    #pragma omp parallel for
    for (auto m_ = 0; m_ < m; ++m_) {
      auto& subspace_graph = sphere_graph[m_];
      for (auto i = 0; i < nclusters; ++i) {
        Sph* s1 = &rt_spheres[m_ * nclusters + i];
        for (auto j = i + 1; j < nclusters; ++j) {
          Sph* s2 = &rt_spheres[m_ * nclusters + i];
          const T partial_dist[3] = {s1->center[0] - s2->center[0], 
                                     s1->center[1] - s2->center[1],
                                     s1->center[2] - s2->center[2]};
          const float dist_sq = partial_dist[0] * partial_dist[0] \
                              + partial_dist[1] * partial_dist[1] \
                              + partial_dist[2] * partial_dist[2]; 
          if (dist_sq < radius_sq) {
            subspace_graph[i].emplace_back(s2, dist_sq, j);
            subspace_graph[j].emplace_back(s1, dist_sq, i);
          }
        }
        sphere_nbr_rank[m_].emplace(subspace_graph[i].size(), i, s1);
      }
    }

    #pragma omp parallel for
    for (auto m_ = 0; m_ < m; ++m_) {
      auto& subspace_graph = sphere_graph[m_];
      auto& nbr_rank_queue = sphere_nbr_rank[m_];
      while (!nbr_rank_queue.empty()) {
        auto [neighbor_count, cluster, vertex] = nbr_rank_queue.top(); nbr_rank_queue.pop();
        if (subspace_graph[cluster].empty()) {
          continue;
        }

        auto& this_neighborhood = rt_cluster_mapping[m_ * nclusters + cluster];
        std::set<int> unique_nbrs(std::begin(this_neighborhood), std::end(this_neighborhood));

        // coalesce points from neighboring spheres into this sphere
        for (auto& [neighbor, dist_sq, nbr_cluster] : subspace_graph[cluster]) {
            auto& nbr_points = rt_cluster_mapping[m_ * nclusters + nbr_cluster];
            for (auto p : nbr_points) {
              unique_nbrs.emplace(p);
            }
            nbr_points.clear();
            neighbor->radius = 0;

            // remove this vertex from its neighbors' adjacency lists
            auto& neighbor_list = subspace_graph[nbr_cluster];
            neighbor_list.erase(
                std::remove_if(std::begin(neighbor_list), std::end(neighbor_list),
                  [&](const auto& p) {
                    return std::get<0>(p) == vertex;
                  }),
                std::end(neighbor_list));
        }
        
        // ensure sphere neighborhood has unique points
        this_neighborhood = {std::begin(unique_nbrs), std::end(unique_nbrs)};

        // increase radius of this sphere
        vertex->radius = 2 * vertex->radius;

        // mark sphere as processed
        subspace_graph[cluster].clear();
      }
    }

    // for (auto j = 0; j < nspheres; ++j) {
    //   for (auto i = 0; i < n; ++i) {
    //     if (assignments[i][j]) {
    //       rt_cluster_mapping[j].push_back(i);
    //     }
    //   }
    // }

    t.Stop();
    runtime = t.Seconds();
    printf("time = %f\n", runtime);
  }
  

  void build_lookup_table(const T *query, 
                          std::vector<std::vector<T>>& lookup_table) {
    // build the lookup table given a query (residual)
    // for each sub-vector space
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
      // i-th sub-vector of the query
      auto q_i = query + i * sub_dim;
      // for each centroid
      auto c_i = centroids[i];
      for (int j = 0; j < nclusters; j++) {
        // j-th centroid in i-th sub-space
        auto c_ij = c_i + j * sub_dim;
        lookup_table[i][j] = compute_distance_squared<T>(sub_dim, q_i, c_ij);
      }
    }
  }


  float quantized_distance(size_t vec_id, 
                           std::vector<std::vector<T>>& lookup_table) {
    float distance = 0;
    // for each sub-vector space
    for (int j = 0; j < m; j++) {
      auto centroid = codebook[vec_id][j];
      distance += lookup_table[j][centroid];
    }
    return distance;
  }


  void search_default(const T* query, int k, int nprobe, int* results) {
    // find closest `nprobe` coarse centroids
    vector<pair<float, int>> top_coarse(nlist);
    for (auto i = 0; i < nlist; ++i) {
      auto dist = compute_distance_squared(dim, query, coarse_centroids[i].first);
      top_coarse[i] = {dist, i};
    }
    sort(begin(top_coarse), end(top_coarse));

    // compute residuals
    T residuals[nprobe * dim] = {0};
    for (auto i = 0; i < nprobe; ++i) {
      auto c_id = top_coarse[i].second;
      memcpy(&residuals[i * dim], query, sizeof(T) * dim);
    }

    // get nearest neighbors
    priority_queue<pair<float, int>> S;
    for (auto i = 0; i < nprobe; ++i) {
      vector<vector<T>> lookup_table(m, vector<T>(nclusters, 0));
      build_lookup_table(&residuals[i * dim], lookup_table);
      auto c_id = top_coarse[i].second;
      auto c_size = coarse_centroids_size[c_id];
      auto c_data = coarse_centroids[c_id].second;

      for (auto j = 0; j < c_size; ++j) {
        auto dist = quantized_distance(c_data[j], lookup_table);
        S.emplace(dist, c_data[j]);
        if ((int)S.size() > k) {
          S.pop();
        }
      }
    }

    for (auto i = 0; i < k; ++i) {
      results[i] = S.top().second; S.pop();
    }
  }

  void print_lookup_table(vector<vector<T>>& lookup_table) {
    cout << "Lookup table:" << endl;
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < nclusters; ++j) {
        cout << lookup_table[i][j] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  ~CoreRT() {}

  static constexpr float factors[64] = \
    {226.91, 226.292, 234.105, 245.577, 279.63, 236.516, 231.948, 269.431,
    274.614, 244.002, 235.553, 258.38, 243.939, 237.857, 229.811, 229.819,
    244.322, 226.982, 252.21, 246.903, 265.966, 238.008, 231.935, 249.658, 
    278.304, 241.357, 236.966, 259.187, 245.247, 245.449, 244.663, 229.863, 
    238.673, 245.904, 235.468, 238.296, 266.595, 246.564, 229.863, 245.392, 
    275.224, 245.247, 239.019, 254.136, 239.708, 236.212, 248.244, 244.125, 
    237.346, 247.491, 225.754, 225.657, 276.957, 235.85, 229.142, 265.548, 
    285.272, 237.186, 252.723, 263.139, 240.983, 220.048, 237.626, 236.326};
  
  static constexpr int tile_width = 8;
  static constexpr size_t bvh_invalid_id = std::numeric_limits<size_t>::max();
  static constexpr size_t bvh_stack_size = 1024;
};

Transform compute_homography(const std::vector<Eigen::Vector3f>& src, const std::vector<Eigen::Vector3f>& dst) {
  // compute a homography from `nh` point correspondences
  const auto nh = src.size();

  Eigen::MatrixXf A(2 * nh, 2 * nh);
  Eigen::MatrixXf B(2 * nh, 1);

  for (auto i = 0; i < nh; ++i) {
    auto x = src[i](0);
    auto y = src[i](1);
    auto z = src[i](2);
    auto xp = dst[i](0);
    auto yp = dst[i](1);
    auto zp = dst[i](2);

    // even
    A(2 * i, 0) = x;
    A(2 * i, 1) = y;
    A(2 * i, 2) = z;
    A(2 * i, 3) = 0;
    A(2 * i, 4) = 0;
    A(2 * i, 5) = 0;
    A(2 * i, 6) = -x * xp;
    A(2 * i, 7) = -y * xp;

    // odd
    A(2 * i + 1, 0) = 0;
    A(2 * i + 1, 1) = 0;
    A(2 * i + 1, 2) = 0;
    A(2 * i + 1, 3) = x;
    A(2 * i + 1, 4) = y;
    A(2 * i + 1, 5) = z;
    A(2 * i + 1, 6) = -x * yp;
    A(2 * i + 1, 7) = -y * yp;

    B(2 * i, 0) = xp;
    B(2 * i + 1, 0) = yp;
  }

  Eigen::MatrixXf x = A.fullPivLu().solve(B);
  Transform H;
  H << x(0), x(1), x(2), \
       x(3), x(4), x(5), \
       x(6), x(7), 1;

  return H;
}
