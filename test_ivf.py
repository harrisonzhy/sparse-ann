import numpy as np
from faiss import index_factory
from faiss import IndexFlatL2
from faiss import IndexIVFPQ

datapath = "data/siftsmall/siftsmall"

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

d = 128
m = 8
nlist = 1

vecs = IndexFlatL2(128)
index = IndexIVFPQ(vecs, d, nlist, m, 8)

print("train index ...")
xt = fvecs_read(datapath + "_base.fvecs")
index.train(xt.astype('float32'))

# print("add xb ...")
# xb = fvecs_read(datapath + "_base.fvecs")
index.add(xt)

xq = fvecs_read(datapath + "_query.fvecs")
gt = ivecs_read(datapath + "_groundtruth.ivecs")

print("search ...")
# index.nprobe = 8  # set w=8
D, I = index.search(xq, 100)

recall_at_100 = (I[:, :100] == gt[:, :1]).sum() / float(xq.shape[0])
print("recall@100: %.3f" % recall_at_100)
