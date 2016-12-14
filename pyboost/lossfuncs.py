import numpy as np


def lossfunc_adtree(rej, left_pos, left_neg, right_pos, right_neg):
    return rej + 2 * (np.sqrt(left_pos * left_neg) + np.sqrt(right_pos * right_neg))
