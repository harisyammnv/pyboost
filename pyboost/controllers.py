"""
.. model:: controllers
"""
import numpy as np
from operator import add

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
    # TODO: may need to broadcast the splitter_node
    return instances.map(
        lambda (y, X, weight): (
            y, X, weight * np.exp(-splitter_node.predict(X, pre_check=False) * y)
        )
    )


def run_adaboost_adtree(y, X, T=10):
    """Train a ADTree using AdaBoost

    :param y: the RDD of the instance labels
    :param X: the RDD of the feature vectors of the instances
    :returns: An array of ADTree nodes with its first element to be the tree root
    """
    # Setup the root of the ADTree
    pos_count = y.filter(lambda y: safe_comp(y, 0.0) > 0).count() + 1
    neg_count = y.filter(lambda y: safe_comp(y, 0.0) < 0).count() + 1
    pred_val = 0.5 * np.log(1.0 * pos_count / neg_count)
    root_node = SplitterNode(0, None, True, TrueCondition())
    root_node.set_predicts(pred_val, 0.0)

    # Setup instances RDD
    instances = y.zip(X).map(lambda (y, X): (y, X, np.exp(-y * pred_val))).cache()

    # Iteratively grow the ADTree
    nodes = [root_node]
    for iteration in range(T):
        # next split
        prt_node, onleft, cond = partition_greedy_split(root_node, instances, lossfunc_adtree)
        # Spark will deep copy the instance, so `prt_node` above is actually a copy
        # rather than a reference
        prt_node = nodes[prt_node.index]
        new_node = SplitterNode(len(nodes), prt_node, onleft, cond)
        # Set the predictions of the new node
        predicts = (
            instances.map(lambda (y, X, w): ((new_node.check(X), safe_comp(y)), w))
                     .filter(lambda ((predict, label), w): predict is not None)
                     .reduceByKey(add)
                     .mapValues(lambda w: w + 1.0)
                     .collectAsMap()
        )
        lpred = 0.5 * np.log(1.0 * predicts[(True, 1)] / predicts[(True, -1)])
        rpred = 0.5 * np.log(1.0 * predicts[(False, 1)] / predicts[(False, -1)])
        new_node.set_predicts(lpred, rpred)
        # add new node to the ADTree
        prt_node.add_child(onleft, new_node)
        nodes.append(new_node)
        # adjust the instances weight
        instances = adaboost_adjust_weight(instances, new_node).cache()
    # Return the ADTree
    return nodes
