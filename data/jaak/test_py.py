#!/usr/bin/env python

import smurff
import matrix_io as mio
import numpy as np
from time import time

#load data
ic50_train = mio.read_matrix("chembl-IC50-346targets-100compounds-train.sdm")
ic50_test = mio.read_matrix("chembl-IC50-346targets-100compounds-test.sdm")
#feat = mio.read_matrix("chembl-IC50-100compounds-feat-dense.ddm")
feat = mio.read_matrix("chembl-IC50-100compounds-feat.sdm")

ic50_threshold = 6.

trainSession = smurff.TrainSession(
                            verbose = 2,
                            priors = ['macau', 'normal'],
                            num_latent=32,
                            num_threads=1,
                            seed=1234,
                            burnin=400,
                            nsamples=200,
                            # Using threshold of 6. to calculate AUC on test data
                            threshold=ic50_threshold)

## using activity threshold pIC50 > 6. to binarize train data
trainSession.addTrainAndTest(ic50_train, ic50_test)
trainSession.addSideInfo(0, feat, noise=smurff.SampledNoise(), direct=False)

start = time()
predictions = trainSession.run()
stop = time()

print("time = %.2f" % (stop - start))
print("RMSE = %.2f" % smurff.calc_rmse(predictions))
print("AUC = %.2f" % smurff.calc_auc(predictions, ic50_threshold))
