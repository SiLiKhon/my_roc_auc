import unittest

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_blobs

from my_roc_auc import my_roc_auc

class MyAUCTest(unittest.TestCase):
    def single_run(self, cl, pred):
        w = np.random.normal(size=len(cl), loc=1., scale=0.2)
        sk_result = roc_auc_score(cl, pred, sample_weight=w)
        my_result = my_roc_auc(cl, pred, weights=w)
        self.assertTrue(np.isclose(sk_result, my_result), "{} != {}".format(my_result, sk_result))
        
        

    def test_without_collisions(self):
        for _ in range(1000):
            X, y = make_blobs(n_features=1, centers=[[0.], [1.]])
            x = X.flatten()
            self.single_run(y, x)

    def test_with_collistions(self):
        for _ in range(1000):
            X, y = make_blobs(n_features=1, centers=[[0.], [1.]])
            x = (X.flatten() * 10).astype(int)
            self.single_run(y, x)

if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
