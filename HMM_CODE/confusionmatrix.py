from numpy import *
import matplotlib.pyplot as plt
from pylab import *

def plotconf(arr, xlab=None, ylab=None, title=None):

    # normalize conf arr (better ways to do this...
    norm_conf = []
    for i in arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    #plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # youwanttext = False
    # if youwanttext:
    #     for i, cas in enumerate(conf_arr):
    #         for j, c in enumerate(cas):
    #             if c > 0:
    #                 plt.text(j - .2, i + .2, c, fontsize=14)

    res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
    cb = fig.colorbar(res)
    if xlab is not None:
        plt.xticks(range(len(arr[0])))
        ax.set_xticklabels(xlab)
    if ylab is not None:
        plt.yticks(range(len(arr)))
        ax.set_yticklabels(ylab)
    if title is not None:
        plt.title(title)
    return plt
    #savefig("confmat.png", format="png")

if __name__ == "__main__":
    conf_arr = [[33, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3], [3, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1], [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0], [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
                [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2], [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0], [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1]]
                #[3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0], [0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 38]]
    conf_arr = [[3, 4, 2], [6, 3, 0]]
    plotconf(conf_arr, xlab=["john", "suj", "han"])
