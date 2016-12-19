"""
.. module:: learners
"""
from numpy.random import randint
from numpy import inf
from operator import itemgetter

from conditions import ThresholdCondition
from utils import safe_comp


def partition_greedy_split(sc, nodes, instances, loss_func, root_index=0, quiet=True):
    """Find the best split from all existing tree nodes and split points

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
        split_index, insts = data
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
            left_insts, right_insts = [], []
            for t in data:
                if node.check(t[2]):
                    left_insts.append(t)
                else:
                    right_insts.append(t)

            # find a best split threshold on this node
            for _loop, insts in enumerate([left_insts, right_insts]):
                onleft = (_loop == 0)
                tot_pos = sum(map(itemgetter(3),
                                  filter(lambda t: safe_comp(t[0], 0.0) > 0,
                                         insts)))
                tot_neg = sum(map(itemgetter(3),
                                  filter(lambda t: safe_comp(t[0], 0.0) < 0,
                                         insts)))
                rej = tot_weight - tot_pos - tot_neg
                left_pos = 0.0
                left_neg = 0.0
                for i in range(len(insts) - 1):
                    y, xi, X, w = insts[i]
                    left_pos += (safe_comp(y, 0.0) > 0) * w
                    left_neg += (safe_comp(y, 0.0) < 0) * w
                    if safe_comp(xi, insts[i + 1][1]):
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

        return (min_score, (r_node, r_onleft, ThresholdCondition(split_index, r_threshold)))

    splits = (
        instances.mapPartitionsWithIndex(extract_data)
                 .map(pgs_find_best_split)
    )
    min_score, (best_node, best_onleft, best_cond) = splits.min(itemgetter(0))
    if not quiet:
        print "Min score:", min_score
    return best_node, best_onleft, best_cond
