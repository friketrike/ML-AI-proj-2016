# COMP 6321 Machine Learning, Fall 2016
# Federico O'Reilly Regueiro - 40012304
# Final project - othello with neural nets

import othello_interface_v2 as oi2

# restoring broken session

oi2.restore_checkpoint()

# since we will call this from a shell script to reduce the mem leak, we do one 25-game batch, otherwise maybe more
num25batches = 1

for _ in range(num25batches):
    for __ in range(25):
        outcome = oi2.play_net(True)
    oi2.save_checkpoint()
    oi2.print_scores()


