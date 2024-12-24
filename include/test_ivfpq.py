import numpy as np
import faiss

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

print("hello")
index = faiss.index_factory(128, "IVF1024,PQ8")

# train the index
print("training ...")
xt = fvecs_read("data/sift/sift_learn.fvecs")
index.train(xt[:100*1024].astype('float32'))

# populate the index
xb = fvecs_read("data/sift/sift_base.fvecs")
index.add(xb)

xq = fvecs_read("data/sift/sift_query.fvecs")
gt = ivecs_read("data/sift/sift_groundtruth.ivecs")

index.nprobe = 8  # set w=8
D, I = index.search(xq, 100)

recall_at_100 = (I[:, :100] == gt[:, :1]).sum() / float(xq.shape[0])
print("recall@100: %.3f" % recall_at_100)
