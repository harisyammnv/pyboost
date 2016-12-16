"""a collection of useful functions"""
from time import time


class Timer:
    def __init__(self):
        self._log = []
        self._last_stop = None
        self._st_time = time()

    def stamp(self, message):
        self._last_stop = time()
        self._log.append("(%.2f s) %s" % (self._last_stop - self._st_time, message))

    def print_last_stamp(self):
        print self._log[-1]

    def print_all(self):
        for s in self._log:
            print s


def safe_comp(a, b=0.0, threshold=1e-8):
    diff = a - b
    if diff < -threshold:
        return -1
    if diff > threshold:
        return 1
    return 0
