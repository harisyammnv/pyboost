from copy import copy


class SplitterNode:
    def __init__(self, prt, on_left, cond):
        self.prt = prt
        self.lpredict = 0.0
        self.rpredict = 0.0
        self.lchild = []
        self.rchild = []
        if self.prt:
            self.preconds = prt.get_preconds(on_left)
        else:
            self.preconds = []
        self.cond = copy(cond)
        return self

    def get_preconds(self, result):
        c = copy(self.cond)
        c.set_result(result)
        return self.preconds + [c]

    def split(self, instance, pre_check=True):
        """Direct an instance to go to the left or right subtree

        return `True` if the instance is in its left child;
        return `False` if the instance is in its right child;
        return `None` if the instance cannot reach this splitter node.
        """
        if not pre_check:
            for c in self.pre_conds:
                if not c.check(instance):
                    return None
        return self.cond(instance)

    def predict(self, instance, pre_check=True):
        """Return the prediction to an instance by this learner"""
        if not pre_check:
            for c in self.pre_conds:
                if not c.check(instance):
                    return 0.0
        if self.cond(instance):
            return self.lpredict
        return self.rpredict

    def set_predict(self, lpredict, rpredict):
        self.lpredict = lpredict
        self.rpredict = rpredict
