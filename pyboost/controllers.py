"""
.. model:: controllers
"""
import numpy as np

from adtree import SplitterNode
from conditions import TrueCondition
from learners import partition_greedy_split
from lossfuncs import lossfunc_adtree
from utils import safe_comp


def adaboost_adjust_weight(instances, splitter_node):
    """Adjust the weights of instances after obtaining a new learner (splitter)

    :param instances: the RDD of the instances, each element is a triplet of (y, X, weight)
    :param splitter: a node in the alternating decision tree, representing the new learner
    :returns: a new RDD of the instances with adjusted weights
    """
    bc_splitter = sc.broadcast(splitter_node)
    return instances.map(
        lambda (y, X, weight): (
            y, X, weight * np.exp(-bc_splitter.value.predict(X, pre_check=False) * y)
        )
    )


def run_adaboost_adtree(y, X, T=10):
    """Train a ADTree using AdaBoost

    :param y: the RDD of the instance labels
    :param X: the RDD of the feature vectors of the instances
    :returns: An array of ADTree nodes with its first element to be the tree root
    """
    # Setup the root of the ADTree
    pos_count = y.filter(lambda y: safe_comp(y, 0.0) > 0).count()
    neg_count = y.filter(lambda y: safe_comp(y, 0.0) < 0).count()
    pred_val = 0.5 * np.log(1.0 * pos_count / neg_count)
    root_node = SplitterNode(None, True, TrueCondition())
    root_node.set_predict(pred_val, 0.0)

    # Setup instances RDD
    instances = y.zip(X).map(lambda (y, X): (y, X, np.exp(-y * pred_val))).cache()

    # Iteratively grow the ADTree
    nodes = [root_node]
    for iteration in range(T):
        # next split
        prt_node, onleft, cond = partition_greedy_split(root_node, instances, lossfunc_adtree)
        new_node = SplitterNode(prt_node, onleft, cond)
        # add new node to the ADTree
        prt_node.add_child(onleft, new_node)
        nodes.append(new_node)
        # adjust the instances weight
        instances = adaboost_adjust_weight(instances, new_node).cache()
    # Return the ADTree
    return nodes
