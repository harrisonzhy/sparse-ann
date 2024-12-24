#include <cstring>
#include <algorithm>
#include <queue>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexPQ.h>
#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/VectorTransform.h>
#include <faiss/utils/distances.h>

#include "utils.hpp"
#include "distance.hpp"

using namespace std;

template <typename T=float, typename CT=uint8_t>
class Quantizer {
private:
  int m; // number of sub-spaces
  int nclusters; // number of clusters in each sub-space
  int dim; // number of dimentions of each vector
  int sub_dim; // number of dimensions of each subvector
  size_t n; // number of vectors in the database
  faiss::IndexFlatL2* quantizer; // trained quantizer (and sub-quantizers)
  faiss::IndexPQ* index; // trained index (for faiss search)
  vector<vector<CT>> codebook;
  vector<T*> centroids; // m * nclusters * sub_dim; PQ centroids

public:
  Quantizer(int m_, int nclusters_, int dim_, size_t n_, const T* data_vectors) :
    m(m_), nclusters(nclusters_), dim(dim_), n(n_), quantizer(nullptr), index(nullptr) {
    assert(dim % m == 0);
    sub_dim = dim / m;
    printf("Quantizer: m=%d, nclusters=%d, dim=%d, n=%lu\n", m, nclusters, dim, n);

    centroids.resize(m, 0);
    codebook.resize(n, std::vector<CT>(m, 0));

    train_centroids_and_build_codebook(data_vectors);
  }

  ~Quantizer() {}

  void train_centroids_and_build_codebook(const T* data) {
    Timer t;
    printf("Train centroid ... \n");

    t.Start();
    quantizer = new faiss::IndexFlatL2(dim);
    // index = new faiss::IndexIVFPQ(quantizer, dim, 1, m, 63 - __builtin_clzll(nclusters));
    index = new faiss::IndexPQ(dim, m, 63 - __builtin_clzll(nclusters));

    index->train(n, data);
    index->add(n, data);

    t.Stop();
    printf("time = %f\n", t.Seconds());

    printf("Build codebook ... \n");
    t.Start();

    // centroid table (m, nclusters, sub_dim)
    for (auto i = 0; i < m; ++i) {
      centroids[i] = index->pq.get_centroids(i, 0);
    }

    #pragma omp parallel for
    for (auto i = 0; i < n; ++i) {
      // for each vector `x_i`
      auto x_i = data + i * dim;
      // for each subspace `j`
      for (auto j = 0; j < m; ++j) {
        float min_dist = __builtin_inff();
        uint16_t best_idx = 0;

        auto sub_xi = x_i + j * sub_dim;
        auto c_j = centroids[j];
        for (auto k = 0; k < nclusters; ++k) {
          auto c_jk = c_j + k * sub_dim;
          auto dist = compute_distance_squared(sub_dim, sub_xi, c_jk);
          if (dist < min_dist) {
            min_dist = dist;
            best_idx = k;
          }
        }
        codebook[i][j] = best_idx;
      }
    }

    t.Stop();
    printf("time = %f\n", t.Seconds());
  }


  void search(const T* query, int k, int* results) {
    std::vector<std::vector<T>> lookup_table(m, std::vector<T>(nclusters, 0));
    build_lookup_table(query, lookup_table);

    std::priority_queue<std::pair<T, int>> S;
    for (size_t i = 0; i < n; ++i) {
      const auto dist = quantized_distance(i, lookup_table);
      S.emplace(dist, i);
      if ((int)S.size() > k) {
        S.pop();
      }
    }
    for (auto i = 0; i < k; ++i) {
      results[i] = S.top().second; S.pop();
    }
  }

  void faiss_search(int nq, const T* queries, int k, int* results) {
    std::vector<float> D(nq * m * nclusters);

    vector<T> dis_tables(nq * m * nclusters);
    faiss::HeapArray<faiss::CMax<T, int>> res = { nq, k, results, D.data() };
    // index->pq.compute_distance_tables(nq, queries, dis_tables.data());
    
    #pragma omp parallel for if (nq > 1)
    for (auto i = 0; i < nq; ++i) {
      const auto query = queries + i * dim;
      T* dis_table = dis_tables.data() + i * m * nclusters;

      for (auto m_ = 0; m_ < m; m_++) {
        faiss::fvec_L2sqr_ny(
                dis_table + m_ * nclusters,
                query + m_ * sub_dim,
                centroids[m_],
                sub_dim,
                nclusters);
      }

      // compute distance tables (LUT)
      auto codes = index->codes.data();
      int* __restrict heap_ids = res.ids + i * k;
      T*   __restrict heap_dis = res.val + i * k;

      faiss::heap_heapify<faiss::CMax<T, int>>(k, heap_dis, heap_ids);
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
      faiss::heap_reorder<faiss::CMax<T, int>>(k, heap_dis, heap_ids);
    }
  }

  void faiss_search_batch(int nq, const T* queries, int k, int* results) {
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);
    index->search(nq, queries, k, D.data(), I.data());
    for (auto i = 0; i < nq * k; ++i) {
      results[i] = (int)I[i];
    }
  }

  // build the lookup table given a query
  void build_lookup_table(const T *query, 
                          std::vector<std::vector<T>>& lookup_table) {
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
        lookup_table[i][j] = compute_distance_squared(sub_dim, q_i, c_ij);
      }
    }
  }

  void compute_residual(int dim, const float* __restrict__ a, 
                                const float* __restrict__ b, 
                                float* __restrict__ r) {
    a = (const float *)__builtin_assume_aligned(a, 32);
    b = (const float *)__builtin_assume_aligned(b, 32);
    r = (float *)__builtin_assume_aligned(r, 32);

    // assume size is divisible by 8
    uint16_t niters = (uint16_t)(dim / 8);
    for (uint16_t j = 0; j < niters; j++) {
      // scope is a[8j:8j+7], b[8j:8j+7]
      if (j+1 < niters) {
        _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
      }
      // load a_vec
      __m256 a_vec = _mm256_load_ps(a + 8 * j);
      // load b_vec
      __m256 b_vec = _mm256_load_ps(b + 8 * j);
      // a_vec - b_vec
      _mm256_store_ps(r + 8 * j, _mm256_sub_ps(a_vec, b_vec));
    }
  }

  // build the lookup table given a query
  //  precompute distances from query to centroids across all subspaces
  void build_lookup_table(const T *query, const T *center,
                          std::vector<std::vector<T>>& lookup_table) {
    T residue[dim];
    compute_residual(dim, query, center, residue);

    // for each sub-vector space
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
      // i-th sub-vector of the query
      // auto q_i = query + i * sub_dim;
      auto q_i = &residue[i * sub_dim];
      // for each centroid
      auto c_i = centroids[i];
      for (int j = 0; j < nclusters; j++) {
        // j-th centroid in i-th sub-space
        auto c_ij = c_i + j * sub_dim;
        lookup_table[i][j] = compute_distance_squared(sub_dim, q_i, c_ij);
      }
    }
  }

  // using this query's LUT, 
  //  accumulate distances from this search point to best centroids across all subspaces
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
};

// divide scene into `nclusters` spheres with big radius
// 

