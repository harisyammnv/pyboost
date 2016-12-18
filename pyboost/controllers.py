"""
.. model:: controllers
"""
import numpy as np
from operator import add

from adtree import SplitterNode
from conditions import TrueCondition
from learners import partition_greedy_split
from lossfuncs import lossfunc_adtree
from updatefuncs import adaboost_update
from updatefuncs import logitboost_update
from utils import safe_comp
from utils import Timer


def _run_adtree(sc, y, X, updatefunc, T, quiet):
    """Train a ADTree

    :param y: the RDD of the instance labels
    :param X: the RDD of the feature vectors of the instances
    :returns: An array of ADTree nodes with its first element to be the tree root
    """
    # Setup the root of the ADTree
    pos_count = y.filter(lambda y: safe_comp(y, 0.0) > 0).count()
    neg_count = y.filter(lambda y: safe_comp(y, 0.0) < 0).count()
    pred_val = 0.5 * np.log(1.0 * pos_count / neg_count)
    root_node = SplitterNode(0, None, True, TrueCondition())
    root_node.set_predicts(pred_val, 0.0)

    # Setup instances RDD
    feature_size = X.first().size
    instances = y.zip(X).map(lambda (y, X): (y, X, 1.0)).repartition(feature_size)
    instances = updatefunc(instances, root_node).cache()

    # Iteratively grow the ADTree
    nodes = [root_node]
    timer = Timer()
    for iteration in range(T):
        timer.stamp("[run_adtree] Iteration " + str(iteration) + " starts.")
        if not quiet:
            print '=' * 3, "Iteration %d" % (iteration + 1), '=' * 3
        # next split
        prt_node, onleft, cond = partition_greedy_split(
            sc, nodes, instances, lossfunc_adtree, quiet=quiet
        )
        timer.stamp("[run_adtree] Found best split.")
        # Spark will deep copy the instance, so `prt_node` above is actually a copy
        # rather than a reference
        prt_node = nodes[prt_node.index]
        new_node = SplitterNode(len(nodes), prt_node, onleft, cond)
        # Set the predictions of the new node
        predicts = (
            instances.map(lambda (y, X, w): ((new_node.check(X, pre_check=False), safe_comp(y)), w))
                     .filter(lambda ((predict, label), w): predict is not None)
                     .reduceByKey(add)
                     .mapValues(lambda w: w)
                     .collectAsMap()
        )

        min_val = min([t for t in predicts.values() if t > 0]) * 0.001
        predicts[(True, 1)] = predicts.get((True, 1), min_val)
        predicts[(True, -1)] = predicts.get((True, -1), min_val)
        predicts[(False, 1)] = predicts.get((False, 1), min_val)
        predicts[(False, -1)] = predicts.get((False, -1), min_val)

        lpred = 0.5 * np.log(1.0 * predicts[(True, 1)] / predicts[(True, -1)])
        rpred = 0.5 * np.log(1.0 * predicts[(False, 1)] / predicts[(False, -1)])
        timer.stamp("[run_adtree] Obtained the predictions of the new split.")
        if not quiet:
            print "Purity (farther from 1.0 is better):",
            print (1.0 * predicts[(True, 1)] / predicts[(True, -1)],
                   1.0 * predicts[(False, 1)] / predicts[(False, -1)])
            print "Predicts (farther from 0.0 is better):", (lpred, rpred)
            print "Split node:", prt_node.index,
            if onleft:
                print "(left)"
            else:
                print "(right)"
            print "Split index and value:", cond.index, cond.val, '\n'
        new_node.set_predicts(lpred, rpred)
        # add new node to the ADTree
        prt_node.add_child(onleft, len(nodes))
        nodes.append(new_node)
        # adjust the instances weight
        instances = updatefunc(instances, new_node).cache()
        timer.stamp("[run_adtree] Instance weights updated.")
    if not quiet:
        print "== Timer Log =="
        timer.print_all()
        print
    # Return the ADTree
    return nodes


def run_adtree_adaboost(sc, y, X, T=10, quiet=True):
    return _run_adtree(sc, y, X, adaboost_update, T, quiet)


def run_adtree_logitboost(sc, y, X, T=10, quiet=True):
    return _run_adtree(sc, y, X, logitboost_update, T, quiet)
