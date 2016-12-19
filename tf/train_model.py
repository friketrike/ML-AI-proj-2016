import pickle
import time
import othello_interface_v2 as oi2
import os.path

ckpt_fname = 'otnet_v2.ckpt'
pckl_fname = 'net_other.pckl'

if os.path.isfile('otnet_v2.ckpt'):
    oi2.restore_checkpoint()
    if os.path.isfile(pckl_fname):
        f = open(pckl_fname, 'rb')
        x = pickle.load(f)
        f.close()
        oi2.score_series = x

for batch in range(200):
    tic = time.time()
    for _ in range(1000):
        oi2.play_net(True)
    print('batch ', batch, ' took ', (time.time()-tic)/60, ' minutes')
    tl = time.localtime()
    print('finished at: ', tl.tm_hour, ':',tl.tm_min)
    oi2.save_checkpoint()
    oi2.save_checkpoint('SavedModels/otnetv2'+str(oi2.score_series.__len__())+'.ckpt')
    f = open(pckl_fname, 'wb')
    pickle.dump(oi2.score_series, f)
    f.close()
