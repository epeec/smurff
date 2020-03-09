
import logging
import numpy as np

import smurff

logging.getLogger().setLevel(logging.INFO)

trainSession = smurff.TrainSession(priors = ["normal", "normal"])
Y = np.array([[1.,2.],[3.,4.]])
print(Y)
trainSession.addTrainAndTest(Y, noise = smurff.FixedNoise())

trainSession.run()

