"""
.. module:: conditions
"""
from utils import safe_comp


class ThresholdCondition:
    """Split instances to two groups using a threshold value on a specific index"""
    def __init__(self, index, split_val):
        """
        :param index: the feature index to be used for splitting
        :param split_val: the threshold for splitting
        """
        self.index = index
        self.val = split_val
        self.result = None

    def check(self, instance):
        """Check whether an instance satisfies the condition"""
        r = (safe_comp(self.instance[self.index], self.val) <= 0)
        if self.result is not None:
            return r == self.result
        return r

    def set_result(self, result):
        """Change the `check` method behavior by setting the expected condition result"""
        self.result = result


class TrueCondition:
    """Always True condition"""
    def __init__(self):
        pass

    def check(self, instance):
        return True
