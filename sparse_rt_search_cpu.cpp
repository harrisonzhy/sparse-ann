#include "utils.hpp"
#include "sparse_rt.hpp"

template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                     const T* queries, const T* data_vectors,
                     int *results, const char *index_file) {
  int num_threads;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("num_threads = %d\n", num_threads);

  int m = dim / 2;
  constexpr int nclusters = 256;
  constexpr int nlist = 1;
  constexpr int nprobe = 1;
  constexpr float radius = 6;

  // good pair 0.91@100 on siftsmall: (radius=8, mconsensus=0.16 * m)
  // good pair 0.91@100 on sift:      (radius=6, mconsensus=0.125 * m)

  CoreRT<T> sparse_rt(nclusters, dim, npoints, data_vectors, nlist, radius);
  std::vector<std::vector<bool>> visited(num_threads, std::vector<bool>(npoints, false)); // internal 1-bit optimization
  
  Timer t;

  t.Start();
  printf("Do search ...\n");

  int qsize_ = qsize;

  // one at a time
  #pragma omp parallel for
  for (auto qid = 0; qid < qsize_; ++qid) {
    auto query = queries + qid * dim;
    auto result = results + qid * k;
    auto tid = omp_get_thread_num();
    sparse_rt.search_one_shot(1, query, k, nprobe, result, visited[tid]);
    // sparse_rt.faiss_search(1, query, k, nprobe, result);
    std::fill(std::begin(visited[tid]), std::end(visited[tid]), false);
  }

  // many at a time
  // sparse_rt.search(qsize, queries, k, nprobe, results);

  t.Stop();
  
  double runtime = t.Seconds();
  auto throughput = qsize_ / runtime;
  auto latency = runtime / qsize_ * 1000.0;
  printf("time = %f\n", runtime);
  printf("avg latency: %f ms/query, throughput: %f queries/sec\n", latency, throughput);
}

template class ANNS<float>;
