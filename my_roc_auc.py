import numpy as np

def my_roc_auc(classes : np.ndarray,
               predictions : np.ndarray,
               weights : np.ndarray = None) -> float:
    if weights is None:
        weights = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(weights)
    assert classes.ndim == predictions.ndim == weights.ndim == 1

    idx = np.argsort(predictions)

    predictions = predictions[idx]
    weights     = weights    [idx]
    classes     = classes    [idx]

    weights_0 = weights * (classes == 0)
    weights_1 = weights * (classes == 1)

    cumsum_0 = weights_0.cumsum()
    return (cumsum_0 * weights_1).sum() / (weights_1 * cumsum_0[-1]).sum()

from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


for _ in range(1000):
    X, y = make_blobs(n_features=1, centers=[[0.], [1.]])
    x = X.flatten()
    w = np.random.normal(size=len(x), loc=1., scale=0.2)
    sk_result = roc_auc_score(y, x, sample_weight=w)
    my_result = my_roc_auc(y, x, weights=w)
    if not np.isclose(sk_result, my_result):
        print(sk_result, my_result)

