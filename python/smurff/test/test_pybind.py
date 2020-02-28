import smurff

c = smurff.Config()
print(c)

c.setPriorTypes([ "normal", "normal" ])

session = smurff.TrainSession()

session.run()

