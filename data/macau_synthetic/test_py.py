#!/usr/bin/env python

import smurff
import matrix_io as mio
import unittest
from os.path import join
import sys
from time import time

global_verbose = 0

class TestMacauSyn_py(unittest.TestCase):
    def get_default_opts(self):
        return {
                "priors"          : [ "macau", "normal" ],
                "num_latent"      : 16,
                "burnin"          : 400,
                "nsamples"        : 200,
                "verbose"         : global_verbose, 
                }

    def get_train_noise(self):
        return smurff.FixedNoise(1.0)

    def get_side_noise(self):
        return smurff.FixedNoise(10.)

    def macau(self, dirname, expected):
        args = self.get_default_opts()

        trainSession = smurff.TrainSession(**args)
        Ytrain = mio.read_matrix(join(dirname, "train.sdm"))
        Ytest = mio.read_matrix(join(dirname, "test.sdm"))
        trainSession.addTrainAndTest(Ytrain, Ytest, self.get_train_noise())

        sideinfo = mio.read_matrix(join(dirname, "rows.ddm"))
        trainSession.addSideInfo(0, sideinfo, self.get_side_noise(), direct = True)
        trainSession.init()

        start = time()
        while trainSession.step(): pass
        rmse = trainSession.getRmseAvg()
        stop = time()
        elapsed = stop - start

        self.assertLess(rmse, expected[0])
        self.assertGreater(rmse, expected[1])
        self.assertLess(elapsed, expected[2])

    def test_macau_100(self):
        self.macau("data_40_30_4_100_global", [ 1.08, 1.0, 240. ])

    def test_macau_20(self):
        self.macau("data_40_30_4_20_global", [ 1.08, 1.0, 240. ])

if __name__ == "__main__":
    global_verbose += sys.argv.count("-v")
    global_verbose += sys.argv.count("--verbose")
    unittest.main()
