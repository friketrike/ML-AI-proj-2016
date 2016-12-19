import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


pckl_fname = 'net_other.pckl'


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]/n


def load_and_display():
    if not os.path.isfile(pckl_fname):
        raise FileNotFoundError
    f = open(pckl_fname, 'rb')
    x = pickle.load(f)
    f.close()
    n = []
    o = []
    for sc in x:
        n.append(sc['net'])
        o.append(sc['opponent'])

    n = np.asarray(n)
    o = np.asarray(o)

    mav = moving_average((n-o), 500)
    plt.plot(mav, 'b', np.zeros(mav.__len__()), 'k--')
    plt.title('scores-moving average, rectangular window, n=500')
    plt.xlabel('epochs')
    plt.ylabel('score moving average')
    plt.show()

    mav2 = moving_average((n-o) > 0, n=250)
    plt.title('games won moving average, rectangular window, n=250')
    plt.xlabel('epochs')
    plt.ylabel('wins moving average')
    plt.plot(mav2, 'b', 0.5*np.ones(mav2.__len__()), 'k--')
    plt.show()