import othello_interface as oi
import numpy as np 

# outcomes = []
# total_losses = []
# log = open('./othellog.txt', 'w')

# restoring broken session
oi.wins = 322
oi.losses = 270
oi.ties = 22

oi.restore_checkpoint()

for x in range(10000):
    outcome, total_loss = oi.play_net(True)
    #outcomes.append(outcome)
    #total_losses.append(total_loss)
    #if x%500 == 0:
    #    print(outcomes, file=log)
    #    print(total_losses, file=log)


