import pickle
import time
import othello_interface_v2 as oi2
import os.path
import glob

ckpt_fname = 'otnet_v2.ckpt'
pckl_fname = 'net_other.pckl'

# use glob since tf can (and has) change the way it names checkpoints
if glob.glob('otnet_v2.ckpt*'):
    oi2.restore_checkpoint()
    if os.path.isfile(pckl_fname):
        f = open(pckl_fname, 'rb')
        x = pickle.load(f)
        f.close()
        oi2.score_series = x
else:
    print('Something failed, couldn\'t find model.')

# If we have a fresh model, create a directory to store checkpoints
if not os.path.isdir('SavedModels'):
    print('Creating folder for saving checkpoints.')
    os.mkdir('SavedModels')

for batch in range(2000):
    tic = time.time()
    for _ in range(100):
        oi2.play_net(True)
    print('batch ', batch, ' took ', (time.time()-tic)/60, ' minutes')
    tl = time.localtime()
    print('finished at: ', tl.tm_hour, ':',tl.tm_min)
    f = open(pckl_fname, 'wb')
    pickle.dump(oi2.score_series, f)
    f.close()
    oi2.save_checkpoint()
    #oi2.save_checkpoint('SavedModels/otnetv2_'+str(oi2.score_series.__len__())+'.ckpt')
    
