
import logging
import numpy as np
import scipy.sparse as sp

import smurff

logging.getLogger().setLevel(logging.INFO)

config = smurff.Config(priors = ["normal", "normal"], verbose = 1)
Y = np.array([[1.,2.],[3.,4.]])
config.addTrainAndTest(Y, sp.csr_matrix(Y), noise = smurff.FixedNoise())

trainSession = smurff.TrainSession(config)
results = trainSession.run()
for r in results:
    print(r)

