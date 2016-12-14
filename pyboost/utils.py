"""a collection of useful functions"""


def safe_comp(a, b=0.0, threshold=1e-8):
    diff = a - b
    if diff < -threshold:
        return -1
    if diff > threshold:
        return 1
    return 0
