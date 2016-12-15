"""
.. model:: updatefuncs
"""
import numpy as np
from operator import itemgetter


def adaboost_update(instances, splitter_node):
    """Adjust the weights of instances after obtaining a new learner (splitter) using AdaBoost

    :param instances: the RDD of the instances, each element is a triplet of (y, X, weight)
    :param splitter: a node in the alternating decision tree, representing the new learner
    :returns: a new RDD of the instances with adjusted weights
    """
    # TODO: may need to broadcast the splitter_node
    raw = instances.map(
        lambda (y, X, weight): (
            y, X, weight * np.exp(-splitter_node.predict(X, pre_check=False) * y)
        )
    ).cache()
    w_sum = 1.0  # raw.map(itemgetter(2)).sum()
    return raw.map(lambda (y, X, weight): (y, X, weight / w_sum))


def logitboost_update(instances, splitter_node):
    """Adjust the weights of instances after obtaining a new learner (splitter) using LogitBoost

    :param instances: the RDD of the instances, each element is a triplet of (y, X, weight)
    :param splitter: a node in the alternating decision tree, representing the new learner
    :returns: a new RDD of the instances with adjusted weights
    """
    # TODO: may need to broadcast the splitter_node
    raw = instances.map(
        lambda (y, X, weight): (
            y, X, weight / (1.0 + np.exp(splitter_node.predict(X, pre_check=False) * y))
        )
    ).cache()
    w_sum = 1.0  # raw.map(itemgetter(2)).sum()
    return raw.map(lambda (y, X, weight): (y, X, weight / w_sum))
