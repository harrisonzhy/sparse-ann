#!bin/bash

pushd ..
make clean
make all
popd

./test_sparsert_cpu.sh

