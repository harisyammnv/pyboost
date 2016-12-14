"""a collection of useful functions"""


def safe_comp(a, b, threshold=1e-8):
    diff = a - b
    if diff < -threshold:
        return -1
    if diff > threshold:
        return 1
    return 0
