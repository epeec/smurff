#!/usr/bin/env python

import smurff
import matrix_io as mio
import numpy as np

#load data



ic50_train = mio.read_matrix("chembl-IC50-346targets-100compounds-train.sdm")
ic50_test = mio.read_matrix("chembl-IC50-346targets-100compounds-test.sdm")
ic50_threshold = 6.

# feat = mio.read_matrix("chembl-IC50-100compounds-feat.sdm")
feat = mio.read_matrix("chembl-IC50-100compounds-feat-dense.ddm")
print("feat: ", feat.shape)
FtF = np.matmul(feat.T, feat)
print("FtF: ", FtF.shape)

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
trainSession.addSideInfo(0, feat, direct=True)

predictions = trainSession.run()
print("RMSE = %.2f" % smurff.calc_rmse(predictions))
print("AUC = %.2f" % smurff.calc_auc(predictions, ic50_threshold))
