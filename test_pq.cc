#include <iostream>
#include <vector>
#include <cassert>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/VectorTransform.h>

template <typename T=float>
class ProductQuantization {
public:
    ProductQuantization(int dimensions, int nclusters, int m)
        : dimensions(dimensions), nclusters(nclusters), m(m) {}

    void build_codebook(const T *data, int num_vectors) {
        // Prepare data for Faiss
        std::vector<float> xb(num_vectors * dimensions);
        for (int i = 0; i < num_vectors; ++i) {
            for (int j = 0; j < dimensions; ++j) {
                xb[i * dimensions + j] = static_cast<float>(data[i * dimensions + j]);
            }
        }

        // Create a Flat index
        faiss::IndexFlatL2 quantizer(dimensions); // Use L2 distance

        faiss::IndexIVFPQ myindex(&quantizer, dimensions, nclusters, m, 8); // 8 bits per sub-quantizer
        myindex.verbose = true;

        // Train the index
        myindex.train(num_vectors, xb.data());

        assert(myindex.invlists != nullptr);

        // Generate codebook
        codebook.resize(nclusters, std::vector<float>(dimensions));

        auto nlist = myindex.invlists->nlist;
        std::cout << "nlist=" << nlist << std::endl;

        for (auto list_no = 0; list_no < nlist; ++list_no) {
            auto ls = myindex.invlists->get_codes(list_no);
            codebook[list_no] = 
        }
    }

    const std::vector<std::vector<float>>& getCodebook() const {
        return codebook;
    }

private:
    int dimensions; // Dimensionality of the data
    int nclusters;     // Number of clusters
    int m;         // Number of subquantizers
    std::vector<std::vector<float>> codebook; // The generated codebook
};

template<typename T=float>
class ProductQuantization {
public:
    ProductQuantization(int dimensions, int nclusters, int m)
        : dimensions(dimensions), nclusters(nclusters), m(m) {}

    void build_codebook(const float *data, int num_vectors) {
        // Prepare data for Faiss
        std::vector<float> xb(num_vectors * dimensions);
        for (int i = 0; i < num_vectors; ++i) {
            for (int j = 0; j < dimensions; ++j) {
                xb[i * dimensions + j] = data[i * dimensions + j];
            }
        }

        // Create a Flat index
        faiss::IndexFlatL2 quantizer(dimensions); // Use L2 distance

        faiss::IndexIVFPQ myindex(&quantizer, dimensions, nclusters, m, 8); // 8 bits per sub-quantizer
        myindex.verbose = true;

        // Train the index
        myindex.train(num_vectors, xb.data());

        printf("done training index\n");
        printf("ntotal=%d\n", myindex.quantizer->ntotal);

        printf("start gen codebook\n");
        // Generate codebook directly from the trained index
        codebook.resize(nclusters, std::vector<float>(dimensions));

        // Get the centroids directly from the index
        const float *centroids = myindex.pq.centroids.data();
        for (int i = 0; i < nclusters; ++i) {
            std::memcpy(codebook[i].data(), centroids + i * dimensions, dimensions * sizeof(float));
            printf("gen codebook %d\n", i);
        }
        printf("done gen codebook\n");
    }

    const std::vector<std::vector<float>>& getCodebook() const {
        return codebook;
    }

private:
    int dimensions;
    int nclusters;
    int m;
    std::vector<std::vector<float>> codebook;
};

int main() {
    // Example data

    int num_vectors = 16384; // Number of data points
    int dimensions = 8;     // Dimensionality of each data point
    int nclusters = 8;       // Number of clusters
    int m = 4;               // Number of subquantizers (subspace)

    float* data = new float[num_vectors * dimensions];
    for (auto i = 0; i < num_vectors; ++i) {
        for (auto d = 0; d < dimensions; ++d) {
            data[i * dimensions + d] = ((double)rand() / RAND_MAX);
        }
    }

    ProductQuantization<float> pq(dimensions, nclusters, m);
    pq.build_codebook(data, num_vectors);

    // Output codebook
    const auto& codebook = pq.getCodebook();
    for (size_t i = 0; i < codebook.size(); ++i) {
        std::cout << "Codebook " << i << ": ";
        for (const auto& val : codebook[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
