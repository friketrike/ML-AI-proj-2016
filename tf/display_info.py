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
    plt.plot(mav2, 'b', 0.5 * np.ones(mav2.__len__()), 'k--')
    plt.title('games won moving average, rectangular window, n=250')
    plt.xlabel('epochs')
    plt.ylabel('wins moving average')
    plt.show()

    plt.plot(np.cumsum((n-o) > 0), 'g', np.cumsum((n-o) == 0), 'b', np.cumsum((n-o) < 0), 'r')
    plt.legend(['wins', 'ties', 'losses'], loc='upper left')
    plt.title('Progression of outcomes over epochs')
    plt.xlabel('epochs')
    plt.ylabel('accumulated outcomes')
    plt.show()

    plt.plot(np.divide((np.cumsum((n - o) > 0))+1, np.cumsum((n-o) < 0)+1), 'g', np.ones(n.__len__()), 'k--')
    plt.title('Ratio of wins over losses over epochs')
    plt.xlabel('epochs')
    plt.ylabel('wins to losses')
    plt.show() 
