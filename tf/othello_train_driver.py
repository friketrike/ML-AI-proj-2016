# COMP 6321 Machine Learning, Fall 2016
# Federico O'Reilly Regueiro - 40012304
# Final project - othello with neural nets

import othello_interface as oi
import numpy as np 

# restoring broken session
oi.wins = 530
oi.losses = 471
oi.ties = 38

oi.restore_checkpoint()

for x in range(10000): # one may dream...
    outcome, total_loss = oi.play_net(True)


