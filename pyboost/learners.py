"""
.. module:: learners
"""
from collections import deque
from numpy.random import randint
from numpy import inf
from operator import itemgetter

from conditions import ThresholdCondition
from utils import safe_comp


def partition_greedy_split(adtree_root, instances, loss_func, quiet=True):
    """Find the best split from all existing tree nodes and split points

    .. seealso:: conditions.ThresholdCondition

    :param adtree_root: root of the alternating decision tree
    :param instances: the RDD of the instances, each element is a triplet of (y, X, weight)
    :param loss_func: the loss function to evaluate the splits
    :returns: a (split_node, split_onleft, split_condition) pair for the best split
    """

    _, X, _ = instances.first()
    feature_size = X.size
    shift = randint(feature_size)

    def pgs_find_best_split(split_index, insts):
        split_index = (split_index + shift) % feature_size
        min_score = inf
        r_node = None
        r_threshold = None
        r_onleft = None

        insts = list(insts)
        tot_weight = sum(map(itemgetter(2), insts))
        sorted_insts = sorted(insts, key=lambda (y, X, weight): X[split_index])
        # TODO: use regular list to replace deque
        queue = deque()
        queue.append((adtree_root, sorted_insts))
        while len(queue):
            node, data = queue.popleft()
            left_insts = [t for t in data if node.check(t[1])]
            right_insts = [t for t in data if not node.check(t[1])]

            # find a best split threshold on this node
            for _loop, insts in enumerate([left_insts, right_insts]):
                onleft = (_loop == 0)
                tot_pos = sum(map(itemgetter(2),
                                  filter(lambda t: safe_comp(t[0], 0.0) > 0,
                                         insts)))
                tot_neg = sum(map(itemgetter(2),
                                  filter(lambda t: safe_comp(t[0], 0.0) < 0,
                                         insts)))
                rej = tot_weight - tot_pos - tot_neg
                left_pos = 0.0
                left_neg = 0.0
                for i in range(len(insts) - 1):
                    y, X, w = insts[i]
                    left_pos += (safe_comp(y, 0.0) > 0) * w
                    left_neg += (safe_comp(y, 0.0) < 0) * w
                    if safe_comp(X[split_index], insts[i + 1][1][split_index]):
                        right_pos = tot_pos - left_pos
                        right_neg = tot_neg - left_neg
                        score = loss_func(rej, left_pos, left_neg, right_pos, right_neg)
                        if safe_comp(score, min_score) < 0:
                            min_score = score
                            r_node = node
                            r_threshold = X[split_index]
                            r_onleft = onleft

            # recursively assess the left and right child split nodes
            for child in node.lchild:
                queue.append((child, left_insts))
            for child in node.rchild:
                queue.append((child, right_insts))

        yield (min_score, (r_node, r_onleft, ThresholdCondition(split_index, r_threshold)))

    assert(instances.count() >= feature_size)
    inst_sets = instances.repartition(feature_size)
    splits = inst_sets.mapPartitionsWithIndex(pgs_find_best_split).cache()
    if not quiet:
        idx_score = []
        for score, (_, _, cond) in splits.collect():
            idx_score.append((cond.index, score))
        print "Score (sorted by index):", list(map(itemgetter(1), sorted(idx_score)))
    # TODO: may need to broadcast the whole ADTree
    _, (best_node, best_onleft, best_cond) = splits.min(itemgetter(0))
    return best_node, best_onleft, best_cond
