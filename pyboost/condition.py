'''
Ideally, this class should be a general class function.
For now, we assume all condition function is simply a numeric comparison on one specific feature.
'''


class Condition:
    def __init__(self, index, split_val):
        self.index = index
        self.val = split_val
        self.result = None

    def check(self, instance):
        r = (self.instance[self.index] <= self.val)
        if self.result is not None:
            return r == self.result
        return r

    def set_result(self, result):
        self.result = result
