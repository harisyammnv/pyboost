"""
.. module:: conditions
"""


class SplitCondition:
    """Split instances to two groups using a threshold value on a specific index
    """
    def __init__(self, index, split_val):
        """
        :param index: the feature index to be used for splitting
        :param split_val: the threshold for splitting
        """
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
