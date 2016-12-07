from copy import copy


class SplitterNode:
    def __init__(self, prt, on_left, cond):
        self.prt = prt
        self.lchild = []
        self.rchild = []
        self.preconds = prt.get_preconds(on_left)
        self.cond = copy(cond)
        return self

    def get_preconds(self, result):
        c = copy(self.cond)
        c.set_result(result)
        return self.preconds + [c]

    def split(self, instance, pre_check=True):
        '''
        return `True` if the instance is in its left child;
        return `False` if the instance is in its right child;
        return `None` if the instance cannot reach this splitter node.
        '''
        if not pre_check:
            for c in self.pre_conds:
                if not c.check(instance):
                    return None
        return self.cond(instance)
