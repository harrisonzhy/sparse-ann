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


using namespace std;

using Scalar = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Sph     = bvh::v2::Sphere<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

using Transform = Eigen::Matrix<float, 3, 3>;

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

  faiss::IndexFlatL2* quantizer; // trained quantizer (and sub-quantizers)
  faiss::IndexPQ* index; // trained index
  vector<vector<CT>> codebook;
  vector<T*> centroids; // m * ncluster * sub_dim; PQ centroids
  vector<vector<T>> lookup_table; // m * ncluster
  vector<pair<T*, const faiss::idx_t*>> coarse_centroids;
  vector<int> coarse_centroids_size;

  Bvh* rt_bvh;
  vector<Sph> rt_spheres;
  vector<vector<int>> rt_cluster_mapping;  // points within a sphere
  vector<vector<int>> rt_adj_centroids;    // centroids within a sphere
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
      rt_bvh(nullptr) {
    
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
    rt_spheres.resize(m * nclusters);
    rt_cluster_mapping.resize(m * nclusters);
    rt_adj_centroids.resize(m * nclusters);
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

    // coarse centroid table (nlist, *)
    // for (auto i = 0; i < nlist; ++i) {
    //   auto ct_i_data = index->invlists->get_ids(i);
    //   auto ct_i_size = index->invlists->list_size(i);

    //   // find center
    //   auto center_i = new T[dim];
    //   memset(center_i, 0, sizeof(T) * dim);
    //   for (auto j = 0; j < ct_i_size; ++j) {
    //     auto vec_id = ct_i_data[j];
    //     auto vec = data + vec_id * dim;
    //     for (auto k = 0; k < dim; ++k) {
    //       center_i[k] += vec[k];
    //     }
    //   }
    //   // normalize accumulation
    //   for (auto k = 0; k < dim; ++k) {
    //     center_i[k] /= (T)ct_i_size;
    //   }
    //   coarse_centroids[i] = { center_i, ct_i_data };
    //   coarse_centroids_size[i] = ct_i_size;
    // }
    t.Stop();
    printf("time = %f\n", t.Seconds());
  }

  void search_one_shot(int nq_, const T* queries, int k, int nprobe, int* results) {
    assert(nprobe == 1);

    constexpr int nq = 1;

    vector<bvh::v2::SmallStack<Bvh::Index, bvh_stack_size>> stacks(nq);
    vector<T> dis_tables(nq * m * nclusters);

    std::vector<float> D(nq * m * nclusters);
    faiss::HeapArray<faiss::CMax<T, int>> res = { nq, k, results, D.data() };

    for (auto qid = 0; qid < nq; ++qid) {
      const auto query = queries + qid * dim;

      T* dis_table = dis_tables.data() + qid * m * nclusters;
      T*    __restrict heap_dis = res.val + qid * k;
      int* __restrict heap_ids = res.ids + qid * k;

      // compute lookup tables
      for (auto j = 0; j < m; ++j) {
        faiss::fvec_L2sqr_ny(
          dis_table + j * nclusters,
          query + j * sub_dim,
          centroids[j],
          sub_dim,
          nclusters);
      }

      faiss::heap_heapify<faiss::CMax<T, int>>(k, heap_dis, heap_ids);

      Timer t;
      for (auto j = 0; j < 2; ++j) {
        t.Start();
        // create ray
        //   z_0 = 0 * radius
        //   z_1 = 3 * radius
        //   z_2 = 6 * radius
        //   ...
        const auto x = query[j * sub_dim];
        const auto y = query[j * sub_dim + 1];
        
        const auto ray = Ray(Vec3(x, y, 0),
                             Vec3(0, 0, 1)
                            //  0
                            //  2 * radius + 1);
        );

        // traverse the BVH
        std::vector<int> cluster_points;
        std::vector<int> tmp_;
        rt_bvh->intersect<true,   /* short circuit intersection tests */
                          true    /* robust intersection test */
                          >(ray, rt_bvh->get_root().index, stacks[qid],
          [&](size_t begin, size_t end) {
            for (size_t s = begin; s < end; ++s) {
              const auto prim_id = rt_bvh->prim_ids[s];
              if (rt_spheres[prim_id].intersect(ray)) {
                const auto& neighborhood = rt_cluster_mapping[prim_id];
                std::set_union(
                    std::begin(cluster_points), std::end(cluster_points),
                    std::begin(neighborhood), std::end(neighborhood),
                    std::back_inserter(tmp_));
                cluster_points = std::move(tmp_);
              }
            }
            return !cluster_points.empty();
          });
        if (cluster_points.empty()) {
          continue;
        }

        // t.Stop();
        // std::cout << "BVH time: " << t.Seconds() << std::endl;
        // std::cout << "num candidates: " << cluster_points.size() << std::endl;

        // t.Start();
        // consider points encapsulated by the spheres
        for (auto p_idx = 0; p_idx < cluster_points.size(); ++p_idx) {
          const auto point = cluster_points[p_idx];
          auto codes = index->codes.data() + point * m;
          const T* dt = dis_table;
          T dis = 0;
          for (auto _ = 0; _ < m; _ += 4) {
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
        // t.Stop();
        // std::cout << "dist accum time: " << t.Seconds() << std::endl;
      }
      faiss::heap_reorder<faiss::CMax<T, int>>(k, heap_dis, heap_ids);
    }
  }

  void search(int nq_, const T* queries, int k, int nprobe, int* results) {
    assert(nprobe == 1);

    constexpr int nq = 1;

    vector<bvh::v2::SmallStack<Bvh::Index, bvh_stack_size>> stacks(nq);
    vector<T> dis_tables(nq * m * nclusters);

    std::vector<float> D(nq * m * nclusters);
    faiss::HeapArray<faiss::CMax<T, int>> res = { nq, k, results, D.data() };

    for (auto qid = 0; qid < nq; ++qid) {
      const auto query = queries + qid * dim;

      T* dis_table = dis_tables.data() + qid * m * nclusters;
      T*    __restrict heap_dis = res.val + qid * k;
      auto* __restrict heap_ids = res.ids + qid * k;

      Timer t;

      // compute lookup tables
      for (auto j = 0; j < m; ++j) {
        faiss::fvec_L2sqr_ny(
          dis_table + j * nclusters,
          query + j * sub_dim,
          centroids[j],
          sub_dim,
          nclusters);
      }

      faiss::heap_heapify<faiss::CMax<T, int>>(k, heap_dis, heap_ids);

      t.Start();
      // traverse the BVH
      for (auto j = 0; j < m; ++j) {
        // create ray
        //   z_0 = 0 * radius
        //   z_1 = 3 * radius
        //   z_2 = 6 * radius
        //   ...
        const auto x = query[j * sub_dim];
        const auto y = query[j * sub_dim + 1];
        const auto ray = Ray(Vec3(x, y, (3 * j) * radius),
                             Vec3(0, 0, 1),
                             0,
                             2 * radius + 1);

        auto prim_id = bvh_invalid_id;
        rt_bvh->intersect<false, /* stop at first intersection */
                          true  /* robust intersection test */
                          >(ray, rt_bvh->get_root().index, stacks[qid],
            [&](size_t begin, size_t end) {
                for (size_t s = begin; s < end; ++s) {
                    const auto j = rt_bvh->prim_ids[s];
                    if (rt_spheres[j].intersect(ray)) {
                      prim_id = j;
                    }
                }
                return prim_id != bvh_invalid_id;
            });
        if (prim_id == bvh_invalid_id) {
          continue;
        }

        const auto& sphere = rt_spheres[prim_id];
        const auto& cluster_points = rt_cluster_mapping[prim_id];

        // #pragma omp parallel for if (cluster_points.size() > 1)
        for (auto p_idx = 0; p_idx < cluster_points.size(); ++p_idx) {
          const auto point = cluster_points[p_idx];
          auto codes = index->codes.data() + point * m;
          const T* dt = dis_table;
          T dis = 0;
          for (auto m_ = 0; m_ < m; m_ += 4) {
            T dism = 0;
            dism  = dt[*codes++];
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
      t.Stop();
      std::cout << "dist accum time: " << t.Seconds() << std::endl;

      faiss::heap_reorder<faiss::CMax<T, int>>(k, heap_dis, heap_ids);
    }
  }


  void faiss_search(int nq, const T* queries, int k, int nprobe, int* results) {
    std::vector<float> D(nq * m * nclusters);
    assert(nprobe == 1);

    vector<T> dis_tables(nq * m * nclusters);
    faiss::HeapArray<faiss::CMax<T, int>> res = { nq, k, results, D.data() };
    index->pq.compute_distance_tables(nq, queries, dis_tables.data());
    
    #pragma omp parallel for if (nq > 1)
    for (auto i = 0; i < nq; ++i) {
      const auto query = queries + i * dim;
      T* dis_table = dis_tables.data() + i * m * nclusters;

      Timer t;
      t.Start();
      for (auto m_ = 0; m_ < m; m_++) {
        faiss::fvec_L2sqr_ny(
                dis_table + m_ * nclusters,
                query + m_ * sub_dim,
                centroids[m_],
                sub_dim,
                nclusters);
      }
      t.Stop();
      // std::cout << "LUT time: " << t.Seconds() << std::endl;

      // compute distance tables (LUT)
      auto codes = index->codes.data();
      int* __restrict heap_ids = res.ids + i * k;
      T*   __restrict heap_dis = res.val + i * k;

      faiss::heap_heapify<faiss::CMax<T, int>>(k, heap_dis, heap_ids);

      t.Start();
      for (auto j = 0; j < n; j++) {
        const T* dt = dis_table;
        T dis = 0;
        for (auto m_ = 0; m_ < m; m_ += 4) {
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
          faiss::heap_replace_top<faiss::CMax<T, int>>(k, heap_dis, heap_ids, dis, j);
        }
      }
      t.Stop();
      std::cout << "dist accum time: " << t.Seconds() << std::endl;

      faiss::heap_reorder<faiss::CMax<T, int>>(k, heap_dis, heap_ids);
    }
  }

  void find_subspace_transforms(const T* data) {
    Timer t;
    printf("Finding subspace transforms...\n");
    t.Start();
    
    std::vector<Transform> forward_transforms(m - 1);

    // list of (m-1) homographies that go from I_{m_-1} to I_{m_}
    #pragma omp parallel for
    for (auto m_ = 1; m_ < m; ++m_) {
      forward_transforms[m_ - 1] = ransac(data, 0, m_);
    }

    // list of m homographies going from I_{m_} to I_0
    for (auto m_ = 1; m_ < m; ++m_) {
      rt_subspace_transforms[m_] = rt_subspace_transforms[m_ - 1] * forward_transforms[m_ - 1].inverse();
    }
    t.Stop();
    printf("time = %f\n", t.Seconds());
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
        std::sort(scores.begin(), scores.end(),
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

      for (auto j = 0; j < nclusters; ++j) {
        auto center = c_i + j * sub_dim;
        // auto x = center[0];
        // auto y = center[1];
        // auto z = (3 * i + 1) * radius + 1;

        // create subspace plane-aligned scene
        Eigen::Vector3f txf_cluster = sub_transform * Eigen::Vector3f(center[0], center[1], 1);
        float x = txf_cluster[0];
        float y = txf_cluster[1];
        float z = 1;
        // float z = (3 * i + 1) * radius + 1;

        rt_spheres[i * nclusters + j] = { Vec3(x, y, z), radius };
      }
    }

    // construct BVH tree on the scene in parallel
    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    vector<BBox> rt_bboxes(rt_spheres.size());
    vector<Vec3> rt_centers(rt_spheres.size());
    executor.for_each(0, rt_spheres.size(), 
      [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
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
    vector<vector<uint8_t>> assignments(n, vector<uint8_t>(rt_spheres.size(), false));
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
          if (dist < radius * radius) {
            auto s_id = j * nclusters + k;
            assignments[i][s_id] = true;
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
    for (auto j = 0; j < rt_spheres.size(); ++j) {
      for (auto i = 0; i < n; ++i) {
        if (assignments[i][j]) {
          rt_cluster_mapping[j].push_back(i);
        }
      }
    }

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
  
  static constexpr size_t bvh_invalid_id = std::numeric_limits<size_t>::max();
  static constexpr size_t bvh_stack_size = 128;
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
