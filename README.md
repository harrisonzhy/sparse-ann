# sparse-ann

## Introduction
The [BigANN challenge](https://big-ann-benchmarks.com/neurips23.html) aims to encourage the development of indexing data structures and search algorithms 
for practical variants of the Approximate Nearest Neighbor (ANN) or Vector search problem on commodity hardware.

This work in progress targets the high-dimensional sparse approximate nearest neighbor search problem using ray tracing primitives built into [FAISS](https://github.com/facebookresearch/faiss).

## Quick Start

Edit `env.sh` to ensure the correct paths point to the libraries in your system, and then:

    $ source env.sh

Put datasets in the `data` directory:

    $ mkdir data

Unpack the `siftsmall` dataset into `data` (see below for source):

    $ cd data
    $ tar -xzvf siftsmall.tar.gz

To run the sparse ray tracing implementation:
    
    $ cd bin
    $ sh ./run.sh

To run other implementations (see `bin`):

    $ make all
    $ cd bin
    $ sh ./run_<impl-name>.sh

For a larger dataset:
    
    $ cd data
    $ wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    $ tar -xzvf sift.tar.gz

## References

1. An Approximate Algorithm for Maximum Inner Product Search over Streaming Sparse Vectors [[Paper]](https://arxiv.org/abs/2301.10622)

2. Billion-scale Similarity Search with GPUs [[Paper]](https://arxiv.org/abs/1702.08734) [[Code]](https://github.com/facebookresearch/faiss)

3. DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
[[Video]](https://www.microsoft.com/en-us/research/video/research-talk-approximate-nearest-neighbor-search-systems-at-scale/)
[[Slides 1]](https://cvpr.thecvf.com/media/cvpr-2023/Slides/18545_SzZdLZD.pdf)
[[Slides 2]](https://people.csail.mit.edu/jshun/6506-s24/lectures/lecture21-2.pdf)

4. Product Quantization for Nearest Neighbor Search [[Paper]](https://ieeexplore.ieee.org/document/5432202)

5. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs [[Paper]](https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf)

6. Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/d0ac28b79816b51124fcc804b2496a36-Paper-Conference.pdf)

7. [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks/tree/main)

8. [ann-benchmarks](https://ann-benchmarks.com/)

## More Datasets

|      Name     |                                       Link                                      |  # Datapoints | Dimensions |  Format |
|:-------------:|:-------------------------------------------------------------------------------:|:-------------:|:----------:|:-------:|
| DEEP 1M       | https://github.com/johnpzh/iQAN_AE/blob/master/scripts/get.deep1m.sh            |     1,000,000 |         96 | float32 |
| DEEP 10M      | https://github.com/johnpzh/iQAN_AE/blob/master/scripts/get.deep10m.sh           |    10,000,000 |         96 | float32 |
| DEEP 100M     | https://github.com/johnpzh/iQAN_AE/blob/master/scripts/get.deep100m.sh          |   100,000,000 |         96 | float32 |
| DEEP 1B       | https://www.tensorflow.org/datasets/catalog/deep1b                              | 1,000,000,000 |         96 | float32 |
|               |                                                                                 |               |            |         |
| SIFT small    | http://corpus-texmex.irisa.fr                                                   |        10,000 |        128 | float32 |
| SIFT 1M       | http://corpus-texmex.irisa.fr                                                   |     1,000,000 |        128 | float32 |
| SIFT 100M     | https://github.com/johnpzh/iQAN_AE/blob/master/scripts/get.sift100m.sh          |   100,000,000 |        128 | float32 |
| SIFT 1B       | http://corpus-texmex.irisa.fr                                                   | 1,000,000,000 |        128 |   uint8 |
|               |                                                                                 |               |            |         |
| GIST1M        | http://corpus-texmex.irisa.fr                                                   |     1,000,000 |        960 | float32 |
|               |                                                                                 |               |            |         |
| YFCC 10M      |                                                                                 |    10,000,000 |        192 |   uint8 |
| YFCC 100M     | https://multimediacommons.wordpress.com/yfcc100m-core-dataset/                  |    99,200,000 |        192 |   uint8 |
| Yandex T2I 1B | https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search | 1,000,000,000 |        200 | float32 |
|               |                                                                                 |               |            |         |
| MS MARCO      | https://microsoft.github.io/msmarco/                                            |     8,841,823 |    ~30,000 | float32 |
| MS SPACEV 1B  | https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B                  | 1,402,020,720 |        100 | float32 |
| MS Turing 30M |                                                                                 |    30,000,000 |        100 | float32 |
