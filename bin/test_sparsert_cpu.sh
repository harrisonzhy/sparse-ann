#!/bin/bash

BIN=sparse_rt_search_cpu
BIN_PATH=~/proj/anns/bin
DATA_PATH=~/proj/anns/data
OUTFILE=output.txt

# DATASET=siftsmall
# NP=0.01
# NQ=100
# DIM=128

DATASET=sift
NP=1
NQ=10000
DIM=128

echo "$BIN_PATH/$BIN $DATA_PATH/$DATASET/$DATASET\_base.fvecs \
      $DATA_PATH/$DATASET/$DATASET\_query.fvecs \
      $DATA_PATH/$DATASET/$DATASET\_groundtruth.ivecs \
      $NP $DIM $NQ $OUTFILE"

$BIN_PATH/$BIN $DATA_PATH/$DATASET/$DATASET\_base.fvecs \
               $DATA_PATH/$DATASET/$DATASET\_query.fvecs \
               $DATA_PATH/$DATASET/$DATASET\_groundtruth.ivecs \
               $NP $DIM $NQ $OUTFILE
