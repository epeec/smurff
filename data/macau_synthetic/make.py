#!/usr/bin/env python

import numpy as np
from scipy import sparse
import scipy.io as sio
import argparse
import os
import itertools
import matrix_io as mio
from sklearn import preprocessing
from smurff import make_train_test


# dense -> sparse
#  or
# sparse -> even sparser
def sparsify(A, density):
    if sparse.issparse(A):
        (I, J, V) = sparse.find(A)
    else:
        V = A.reshape(A.size)
        (I, J) = np.indices(A.shape)
        I = I.reshape(A.size)
        J = J.reshape(A.size)

    size = V.size
    num = int(size * density)
    idx = np.random.choice(size, num, replace=False)

    return sparse.coo_matrix((V[idx], (I[idx], J[idx])), shape = A.shape)

def gen_matrix(shape, num_latent, density = 1.0 ):
    X = np.random.normal(size=(shape[0],num_latent))
    W = np.random.normal(size=(shape[1],num_latent))
    Y = np.dot(X, W.transpose()) + np.random.normal(size=shape)
    Y = sparsify(Y, density)
    return Y, X ,W

def write_matrix(dirname, filename, A):
    fname = os.path.join(dirname, filename)
    if sparse.issparse(A):
        mio.write_sparse_float64(fname + ".sdm", A)
    else:
        mio.write_dense_float64(fname + ".ddm", A)

def gen_and_write(shape, num_latent, density, center = "none"):
    Y, X, W = gen_matrix(shape,num_latent,density)
    Ytrain, Ytest = make_train_test(Y, 0.8)
    shape_str = "_".join(map(str,shape))
    dirname = "data_%s_%d_%d_%s" % (shape_str, num_latent, int(density * 100), center)

    if os.path.exists(dirname):
        print("Already exists: %s. Skipping" % dirname)
        return

    print("%s..." % dirname)
    os.makedirs(dirname)

    # PAY ATTENTION TO AXIS ORDER
    if (center == "row"):
        Y = preprocessing.scale(Y, axis = 0, with_std=False)
    elif (center == "col"):
        Y = preprocessing.scale(Y, axis = 1, with_std=False)
    elif (center == "global"):
        Y.data = Y.data - np.mean(Y.data)
    else:
        assert center == "none"

    write_matrix(dirname, "train", Ytrain)
    write_matrix(dirname, "test",  Ytest)
    write_matrix(dirname, "rows",  X)
    write_matrix(dirname, "cols",  W)

def gen_matrix_tests():
    # shape = [2000,100]
    shape = [40,30]
    num_latent = 4
    center = "global"
    for density in (1, .2):
        gen_and_write(shape, num_latent,density, center)

if __name__ == "__main__":
    gen_matrix_tests()
