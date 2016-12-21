"""
.. module:: learners
"""
import numpy as np
from numpy.random import randint
from numpy import inf
from operator import itemgetter

from conditions import ThresholdCondition
from utils import safe_comp


def find_best_split(insts, index):
    min_score = inf
    r_node = None
    r_threshold = None
    r_onleft = None

    tot_weight = sum(map(itemgetter(3), insts))
    queue = []
    queue.append((root_index, insts))
    ptr = 0
    while ptr < len(queue):
        node_index, data = queue[ptr]
        node = bc_nodes.value[node_index]
        ptr = ptr + 1
        check = node.check
        in_left = [check(t[2]) for t in data]
        left_insts = [t for t, left in zip(data, in_left) if left]
        right_insts = [t for t, left in zip(data, in_left) if not left]

        # find a best split threshold on this node
        for _loop, insts in enumerate([left_insts, right_insts]):
            if not insts:
                continue
            onleft = (_loop == 0)
            tot_w = [(t[3], 0.0) if safe_comp(t[0], 0.0) > 0 else (0.0, t[3]) for t in insts]
            tot_pos, tot_neg = np.sum(tot_w, axis=0)
            rej = tot_weight - tot_pos - tot_neg
            left_pos = 0.0
            left_neg = 0.0
            for i in range(len(insts) - 1):
                y, xi, X, w = insts[i]
                xi_next = insts[i + 1][1]
                if safe_comp(y, 0.0) > 0:
                    left_pos += w
                else:
                    left_neg += w
                if safe_comp(xi, xi_next):
                    right_pos = tot_pos - left_pos
                    right_neg = tot_neg - left_neg
                    score = loss_func(rej, left_pos, left_neg, right_pos, right_neg)
                    if safe_comp(score, min_score) < 0:
                        min_score = score
                        r_node = node
                        r_threshold = 0.5 * (xi + xi_next)
                        r_onleft = onleft

        # recursively assess the left and right child split nodes
        queue += (
            [(child, left_insts) for child in node.lchild] +
            [(child, right_insts) for child in node.rchild]
        )

    return (min_score, (r_node, r_onleft, ThresholdCondition(index, r_threshold)))


def partition_greedy_split(sc, nodes, instances, loss_func,
                           repartition=False, root_index=0, quiet=True):
    """Find the best split from all existing tree nodes and split points
    after partitioning the data into as many groups as the number of features.

    .. seealso:: conditions.ThresholdCondition

    :param sc: the SparkContext object
    :param nodes: an array of alternating decision tree nodes, the root index is 0 by default
    :param instances: the RDD of the instances, each element is a triplet of (y, X, weight)
    :param loss_func: the loss function to evaluate the splits
    :returns: a (split_node, split_onleft, split_condition) pair for the best split
    """

    _, X, _ = instances.first()
    feature_size = X.size
    shift = randint(feature_size)
    bc_nodes = sc.broadcast(nodes)

    def extract_data(index, insts):
        split_index = (index + shift) % feature_size
        yield (
            split_index,
            sorted([(t[0], t[1][split_index], t[1], t[2]) for t in insts], key=itemgetter(1))
        )

    def pgs_find_best_split(data):
        index, insts = data
        return find_best_split(index, insts)

    if repartition:
        instances = instances.repartition(feature_size)
    splits = (
        instances.mapPartitionsWithIndex(extract_data)
                 .map(pgs_find_best_split)
    )
    min_score, (best_node, best_onleft, best_cond) = splits.min(itemgetter(0))
    if not quiet:
        print "Min score:", min_score
    return best_node, best_onleft, best_cond


def full_greedy_split(sc, nodes, instances, loss_func, repartition=False, root_index=0, quiet=True):
    """Find the best split from all existing tree nodes and split points
    by running over the entire data

    .. seealso:: conditions.ThresholdCondition

    :param sc: the SparkContext object
    :param nodes: an array of alternating decision tree nodes, the root index is 0 by default
    :param instances: the RDD of the instances, each element is a triplet of (y, X, weight)
    :param loss_func: the loss function to evaluate the splits
    :returns: a (split_node, split_onleft, split_condition) pair for the best split
    """

    _, X, _ = instances.first()
    feature_size = X.size
    bc_nodes = sc.broadcast(nodes)

    data = instances.collect()
    splits = []
    for idx in range(feature_size):
        insts = sorted(list(map(lambda (y, X, w): (y, X[idx], X, w), data)), key=itemgetter(1))
        splits.append(find_best_split(insts, idx))
    min_score, (best_node, best_onleft, best_cond) = min(splits, key=itemgetter(0))
    if not quiet:
        print "Min score:", min_score
    return best_node, best_onleft, best_cond
