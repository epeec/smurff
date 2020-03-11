
import logging
import numpy as np

import smurff

logging.getLogger().setLevel(logging.INFO)

config = smurff.Config(priors = ["normal", "normal"])
Y = np.array([[1.,2.],[3.,4.]])
config.addTrainAndTest(Y, noise = smurff.FixedNoise())

trainSession = smurff.TrainSession(config)
trainSession.run()

