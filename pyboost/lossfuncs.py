import numpy as np


def lf_adtree(wleft, yleft, wright, yright, sum_weight):
    def pn_weight(ws, ys):
        pos, neg = 0.0, 0.0
        for w, y in zip(ws, ys):
            if y == 1.0:
                pos += w
            else:
                neg += w
        return pos, neg

    r = sum_weight - np.sum(wleft) - np.sum(wright)
    pleft, nleft = pn_weight(wleft, yleft)
    pright, nright = pn_weight(wright, yright)
    r = r + 2 * (np.sqrt(pleft * nleft) + np.sqrt(pright * nright))
    return r
