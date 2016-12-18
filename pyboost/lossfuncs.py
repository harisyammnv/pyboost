import numpy as np


def lossfunc_adtree(rej, left_pos, left_neg, right_pos, right_neg):
    t = np.array([rej, left_pos, left_neg, right_pos, right_neg])
    rej, left_pos, left_neg, right_pos, right_neg = t / t.sum()
    return rej + 2 * (np.sqrt(left_pos * left_neg) + np.sqrt(right_pos * right_neg))
