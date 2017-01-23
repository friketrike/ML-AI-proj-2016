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
    e = []
    for sc in x:
        n.append(sc['net'])
        o.append(sc['opponent'])
        e.append(sc['error'])

    n = np.asarray(n)
    o = np.asarray(o)

    window_length = n.__len__() // 10
    mav = moving_average((n-o), window_length)
    plt.plot(mav, 'b', np.zeros(mav.__len__()), 'k--')
    plt.title('scores-moving average, rectangular window, n='+str(window_length))
    plt.xlabel('epochs')
    plt.ylabel('score moving average')
    plt.show()

    mav2 = moving_average((n-o) > 0, window_length)
    plt.plot(mav2, 'b', 0.5 * np.ones(mav2.__len__()), 'k--')
    plt.title('games won moving average, rectangular window, n='+str(window_length))
    plt.xlabel('epochs')
    plt.ylabel('wins moving average')
    plt.show()

    mav3 = moving_average(e, window_length)
    plt.plot(mav3, 'r')
    plt.title('Errors moving average, rectangular window, n='+str(window_length))
    plt.xlabel('epochs')
    plt.ylabel('errors noving average')
    plt.show()

    plt.plot(np.divide((np.cumsum((n - o) > 0))+1, np.cumsum((n-o) < 0)+1), 'g', np.ones(n.__len__()), 'k--')
    #plt.ylim([0.85, 1.02])
    plt.title('Ratio of wins over losses over epochs')
    plt.xlabel('epochs')
    plt.ylabel('wins to losses')
    plt.show() 
