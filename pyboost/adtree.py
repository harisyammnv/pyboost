from copy import copy


def run_tree(index, nodes, instance, max_index=None, quiet=True):
    """Iterate an instance in the subtree rooted at this splitter node.

    .. note:: The pre-conditions are assumed to be satisfied.
    """
    if max_index is not None and index > max_index:
        return 0.0
    node = nodes[index]
    if node.cond(instance):
        score = node.lpredict
        child = node.lchild
        if not quiet:
            print "go to left child, score:", score
    else:
        score = node.rpredict
        child = node.rchild
        if not quiet:
            print "go to right child, score:", score
    for c in child:
        score += run_tree(c, nodes, instance, max_index, quiet)
    return score


class SplitterNode:
    def __init__(self, index, prt, on_left, cond):
        self.index = index
        self.prt = prt
        self.lpredict = 0.0
        self.rpredict = 0.0
        self.lchild = []
        self.rchild = []
        if self.prt:
            self._pre_conds = prt._get_preconds(on_left)
        else:
            self._pre_conds = []
        self.cond = copy(cond)

    def _get_preconds(self, result):
        c = copy(self.cond)
        c.set_result(result)
        return self._pre_conds + [c]

    def check(self, instance, pre_check=True):
        """Check if an instance should go to the left or right subtree

        return `True` if the instance is in its left child;
        return `False` if the instance is in its right child;
        return `None` if the instance cannot reach this splitter node.
        """
        if not pre_check:
            for c in self._pre_conds:
                if not c.check(instance):
                    return None
        return self.cond(instance)

    def predict(self, instance, pre_check=True):
        """Return the prediction to an instance by this learner"""
        if not pre_check:
            for c in self._pre_conds:
                if not c.check(instance):
                    return 0.0
        if self.cond(instance):
            return self.lpredict
        return self.rpredict

    def set_predicts(self, lpredict, rpredict):
        """Set the prediction value of this splitter node"""
        self.lpredict = lpredict
        self.rpredict = rpredict

    def add_child(self, onleft, new_node_index):
        """Add a child under this splitter node"""
        if onleft:
            self.lchild.append(new_node_index)
        else:
            self.rchild.append(new_node_index)
