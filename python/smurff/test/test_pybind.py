import smurff

c = smurff.Config()
print(c)

c.setPriorTypes([ "normal", "normal" ])

trainSession = smurff.TrainSession()

trainSession.run()

