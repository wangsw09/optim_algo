import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.random as npr
import numpy.linalg as npla

import gd

def test_linear_pst_ridge():
    n = 10
    p = 5
    npr.seed(124)
    y = npr.binomial(1, 0.5, size=n) * 2 - 1.0
    X = npr.normal(size=(n, p))
    lam = 2.0
    correct_ans = np.array([0.0, 0.04535591, 0.13250916, 0.28175602, 0.64868765])
    proj_gd_ans = gd.model.linear_pst_ridge(y, X, lam=lam,
            max_iter=10000, tol=1e-9, algo="proj_gd")

    assert npla.norm(correct_ans - proj_gd_ans) < 1e-5

    n = 10
    p = 15
    npr.seed(124)
    y = npr.binomial(1, 0.5, size=n) * 2 - 1.0
    X = npr.normal(size=(n, p))
    lam = 2.0
    correct_ans = np.array([0.0, 0.0, 0.0, 0.0, 0.13922367, 0.0, 0.32567969,
        0.0, 0.0, 0.0, 0.49177005, 0.13575527, 0.00212733, 0.15745742, 0.0])
    proj_gd_ans = gd.model.linear_pst_ridge(y, X, lam=lam,
            max_iter=10000, tol=1e-9, algo="proj_gd")

    assert npla.norm(correct_ans - proj_gd_ans) < 1e-5

