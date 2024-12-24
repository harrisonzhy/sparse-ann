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

template <typename T=float, typename CT=uint16_t>
class CoreRT {
private:
  int m; // number of sub-spaces
  int nclusters; // number of clusters in each sub-space
  int dim; // number of dimentions of each vector
  int sub_dim; // number of dimensions of each subvector
  size_t n; // number of vectors in the database

  int nlist;
  float radius;

  faiss::IndexFlatL2* quantizer; // trained quantizer (and sub-quantizers)
  faiss::IndexPQ* index; // trained index (for faiss search)
  vector<vector<CT>> codebook;
  vector<T*> centroids; // m * ncluster * sub_dim; PQ centroids
  vector<vector<T>> lookup_table; // m * ncluster


  Bvh*              rt_bvh;
  vector<vector<T>> rt_data_vectors;
  vector<pair<int, vector<T>>> rt_coarse_centroids;
  vector<int>       rt_coarse_cluster_sizes;
  vector<T>         rt_data_labels;
  vector<vector<T>> rt_queries;
  vector<int>         rt_gt;
  vector<vector<vector<vector<T>>>> rt_cb_entry;
  vector<vector<vector<T>>> rt_cb_entry_labels;
  vector<Sph>       rt_spheres;

  static constexpr int rt_nqueries = 10000; // max queries
  static constexpr int rt_topk = 100;       // max k in top-k
  static constexpr int rt_pq_entry = 32;
  static constexpr int rt_pq_subdim = 2;    // (x, y) dim pairs -> divide dim by 2
  int rt_pq_dim = dim / rt_pq_subdim;

public:
  CoreRT(int m_, int nclusters_, int dim_, size_t n_, const T* data_vectors, int nlist_, float radius_, string path) 
    : m(m_), 
      nclusters(nclusters_),
      dim(dim_), 
      n(n_),
      nlist(nlist_),
      radius(radius_),
      quantizer(nullptr), 
      index(nullptr),
      rt_bvh(nullptr) {
    
    assert(dim % m == 0);
    assert(nlist > 0);
    sub_dim = dim / m;

    printf("CoreRT: m=%d, nclusters=%d, dim=%d, n=%lu\n", m, nclusters, dim, n);
    printf("set data path as: %s\n", path.c_str());

    centroids.resize(m, 0);
    codebook.resize(n, std::vector<CT>(m, 0));
    lookup_table.resize(m, std::vector<T>(nclusters, 0));

    {
        // rt setup
        rt_data_vectors.resize(n, vector<T>(dim, 0));
        rt_coarse_centroids.resize(nlist);
        rt_data_labels.resize(n, 0);
        rt_queries.resize(rt_nqueries, vector<T>(dim, 0));
        rt_gt.resize(rt_nqueries * rt_topk);
        rt_coarse_cluster_sizes.resize(nlist);

        rt_cb_entry.resize(nlist, vector<vector<vector<T>>>(rt_pq_dim, vector<vector<T>>(rt_pq_entry, vector<T>(rt_pq_subdim))));
        
        printf("Reading in data...\n");
        read_data(path);

        printf("Constructing BVH...\n");
        construct_BVHPQ();
    }
  }

  void search(const T* query, int k, int nl, int* results) {
    assert(nl == 1);

    vector<float> D(nlist);
    for (auto i = 0; i < nlist; ++i) {
        D[i] = compute_distance_squared(dim, query, rt_coarse_centroids[i].second.data());
    }
    
    sort(begin(rt_coarse_centroids), end(rt_coarse_centroids),
        [D](const auto& cen1, const auto& cen2) {
            return D[cen1.first] < D[cen2.first];
        });

    for (auto i = 0; i < nl; ++i) {
        vector<Ray> rays(rt_pq_dim);
        for (auto j = 0; j < rt_pq_dim; ++j) {
            auto x = query[2 * j];
            auto y = query[2 * j + 1];
            rays[j] = { Vec3(0, 0, 0), Vec3(x, y, 2 * j), 0, 10000 };
        }

        // traverse the BVH and get the u, v coordinates of the closest intersection.
        for (auto ray : rays) {
            auto prim_id = bvh_invalid_id;
            Scalar u, v;
            bvh::v2::SmallStack<Bvh::Index, bvh_stack_size> stack;
            rt_bvh->intersect<false, bvh_use_robust_traversal>(ray, rt_bvh->get_root().index, stack,
                [&] (size_t begin, size_t end) {
                    for (size_t i = begin; i < end; ++i) {
                        if (auto hit = rt_spheres[rt_bvh->prim_ids[i]].intersect(ray)) {
                            prim_id = i;
                            std::tie(u, v) = *hit;
                        }
                    }
                    return prim_id != bvh_invalid_id;
                });
            if (prim_id != bvh_invalid_id) {
                std::cout
                    << "Intersection found: " << "primitive: " << prim_id << ", "
                                              << "distance: "  << ray.tmax
                                              << "u = " << u << ", "
                                              << "v = " << v
                                              << std::endl;
            } else {
                std::cout << "No intersection found" << std::endl;
            }
        }
    }
  }

  void search_batched(int qsize, const T* queries, int k, int nl, int* results) {
    assert(nl == 1);

    vector<float> D(qsize * nlist);
    for (auto q = 0; q < qsize; ++q) {
        for (auto i = 0; i < nlist; ++i) {
            D[q * nlist + i] = compute_distance_squared(dim, queries + dim * q, rt_coarse_centroids[i].second.data());
        }
        sort(begin(rt_coarse_centroids), end(rt_coarse_centroids),
            [&](const auto& cen1, const auto& cen2) {
                return D[q * nlist + cen1.first] < D[q * nlist + cen2.first];
            });
    }

    for (auto i = 0; i < nl; ++i) {
        vector<Ray> rays(qsize * rt_pq_dim);
        for (auto q = 0; q < qsize; ++q) {
            auto query = queries + dim * q;
            for (auto j = 0; j < rt_pq_dim; ++j) {
                auto x = query[2 * j];
                auto y = query[2 * j + 1];
                rays[q * rt_pq_dim + j] = { Vec3(0, 0, 0), Vec3(x, y, 2 * j), 0, 10000 };
            }
        }

        // traverse the BVH and get the u, v coordinates of the closest intersection.
        for (auto ray : rays) {
            auto prim_id = bvh_invalid_id;
            Scalar u, v;
            bvh::v2::SmallStack<Bvh::Index, bvh_stack_size> stack;
            rt_bvh->intersect<false, bvh_use_robust_traversal>(ray, rt_bvh->get_root().index, stack,
                [&] (size_t begin, size_t end) {
                    for (size_t i = begin; i < end; ++i) {
                        if (auto hit = rt_spheres[rt_bvh->prim_ids[i]].intersect(ray)) {
                            prim_id = i;
                            std::tie(u, v) = *hit;
                        }
                    }
                    return prim_id != bvh_invalid_id;
                });
            if (prim_id != bvh_invalid_id) {
                std::cout
                    << "Intersection found: " << "primitive: " << prim_id << ", "
                                              << "distance: "  << ray.tmax
                                              << " u = " << u << ", "
                                              << " v = " << v
                                              << std::endl;
            } else {
                std::cout << "No intersection found" << std::endl;
            }
        }
    }
  }









  void construct_BVHPQ() {
    for (auto i = 0; i < nlist; ++i) {
        build_coarse_core(i);
    }
  }

  void build_coarse_core(int list_num) {
    auto ncircles_per_dimpair = rt_pq_entry;
    auto ndimpair = dim / rt_pq_subdim;
    auto nhitable = ncircles_per_dimpair * ndimpair;

    rt_spheres.resize(nhitable);
    for (auto i = 0; i < ndimpair; ++i) {
        auto radius_ = radius * factors[i];
        std::cout << "radius: " << radius << std::endl;
        for (auto j = 0; j < ncircles_per_dimpair; ++j) {
            float x = rt_cb_entry[list_num][i][j][0];
            float y = rt_cb_entry[list_num][i][j][1];
            std::cout << "Create sphere (" << x << ", " << y << ", " << 2 * i + 1 << ", " << radius_ << ")" << std::endl;
            rt_spheres[i * ncircles_per_dimpair + j] = { Vec3(x, y, 2 * i + 1), radius_ };
        }
    }

    // construct BVH tree on the scene, in parallel
    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    std::vector<BBox> bboxes(rt_spheres.size());
    std::vector<Vec3> centers(rt_spheres.size());
    executor.for_each(0, rt_spheres.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            bboxes[i]  = rt_spheres[i].get_bbox();
            centers[i] = rt_spheres[i].get_center();
        }
    });

    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
    rt_bvh = new Bvh(bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config));
  }

  void read_data(string& path) {
    {
        // read data vectors
        string datapath = path + "search_points";
        std::cout << "Reading data `" << datapath << "`" << std::endl;
        std::ifstream f(datapath, std::ios::in);
        for (size_t i = 0; i < n; ++i) {
            for (auto j = 0; j < dim; ++j) {
                f >> rt_data_vectors[i][j];
            }
        }
        f.close();
    }
    {
        // read coarse cluster centroids
        string datapath = path + "cluster_centroids";
        std::cout << "Reading data `" << datapath << "`" << std::endl;
        std::ifstream f(datapath, std::ios::in);
        for (auto i = 0; i < nlist; ++i) {
            vector<T> buf(dim);
            for (auto j = 0; j < dim; ++j) {
                f >> buf[j];
            }
            rt_coarse_centroids[i] = {i, buf};
        }
        f.close();
    }
    {   
        // read data labels
        string datapath = path + "search_points_labels";
        std::cout << "Reading data `" << datapath << "`" << std::endl;
        std::ifstream f(datapath, std::ios::in);
        for (size_t i = 0; i < n; ++i) {
            f >> rt_data_labels[i];
            rt_coarse_cluster_sizes[rt_data_labels[i]] += 1;
        }
        f.close();
    }
    {
        // read queries
        string datapath = path + "queries";
        std::cout << "Reading data `" << datapath << "`" << std::endl;
        std::ifstream f(datapath, std::ios::in);
        for (auto i = 0; i < rt_nqueries; ++i) {
            for (auto j = 0; j < dim; ++j) {
                f >> rt_queries[i][j];
            }
        }
        f.close();
    }
    {
        // read ground truth
        string datapath = path + "ground_truth";
        std::cout << "Reading data `" << datapath << "`" << std::endl;
        std::ifstream f(datapath, std::ios::in);
        for (auto i = 0; i < rt_nqueries; ++i) {
            for (auto j = 0; j < rt_topk; ++j) {
                f >> rt_gt[i * rt_topk + j];
            }
        }
        f.close();
    }
    {
        // resize codebook entry label vec
        //    according to `rt_coarse_cluster_sizes`
        rt_cb_entry_labels.resize(nlist);
        for (auto i = 0; i < nlist; ++i) {
            rt_cb_entry_labels[i].resize(rt_pq_dim);
            for (auto j = 0; j < rt_pq_dim; ++j) {
                rt_cb_entry_labels[i][j].resize(rt_coarse_cluster_sizes[i], 0);
            }
        }
    }
    {
        //  codebook entry labels
        string scratch;
        string datapath = path + "parameter_0/codebook_1/";

        for (auto i = 0; i < nlist; ++i) {
            for (auto j = 0; j < rt_pq_dim; ++j) {
                string datapath_ = datapath + "codebook_cluster=" + to_string(i) + "_dim=" + to_string(j);
                // std::cout << "Reading data `" << datapath_ << "`" << std::endl;
                ifstream f(datapath_, std::ios::in);
                for (auto k = 0; k < rt_pq_entry; ++k) {
                    for (auto l = 0; l < rt_pq_subdim; ++l) {
                        f >> rt_cb_entry[i][j][k][l];
                    }
                }

                f >> scratch; // ???
                for (auto m = 0; m < rt_coarse_cluster_sizes[i]; ++m) {
                    f >> rt_cb_entry_labels[i][j][m];
                }
                f.close();
            }
        }
    }
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
  static constexpr size_t bvh_stack_size = 64;
  static constexpr bool bvh_use_robust_traversal = false;
};
