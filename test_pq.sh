rm a.o

g++ test_pq.cc -o a.o \
    -I$HOME/proj/faiss \
    -I$HOME/proj/amd-libm \
    -I$HOME/proj/lapack-3.10.1 \
    -L$HOME/proj/faiss/build/faiss \
    -L$HOME/proj/amd-libm/lib \
    -L$HOME/proj/lapack-3.10.1 \
    -lfaiss \
    -lalm \
    -llapack \
    -lrefblas \
    -lgfortran

export LD_LIBRARY_PATH=$HOME/proj/faiss/build/faiss:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/proj/amd-libm/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/proj/lapack-3.10.1/:$LD_LIBRARY_PATH

./a.o
