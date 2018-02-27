import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.random as npr
import numpy.linalg as npla

import gd

def test_linear_lasso():
    n = 10
    p = 5
    npr.seed(124)
    y = npr.normal(size=n)
    X = npr.normal(size=(n, p))
    lam = 1.0
    correct_ans = np.array([0.0, 0.09724459, -0.26840158, 0.17676027,
        -0.33039065])
    prox_gd_ans = gd.model.linear_lasso(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="prox_gd")
    acc_prox_gd_ans = gd.model.linear_lasso(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="acc_prox_gd")

    assert npla.norm(correct_ans - prox_gd_ans) < 1e-5
    assert npla.norm(correct_ans - acc_prox_gd_ans) < 1e-5

    n = 10
    p = 15
    npr.seed(124)
    y = npr.normal(size=n)
    X = npr.normal(size=(n, p))
    lam = 1.0
    correct_ans = np.array([-0.03864935, -0.70193877, 0.0, 0.0, 0.03342079,
        0.0, -0.05774081, -0.18683645, 0.0, 0.38292185, 0.0, 0.13697894,
        -0.09517525, 0.0, 0.36420183])
    prox_gd_ans = gd.model.linear_lasso(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="prox_gd")
    acc_prox_gd_ans = gd.model.linear_lasso(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="acc_prox_gd")

    assert npla.norm(correct_ans - prox_gd_ans) < 1e-5
    assert npla.norm(correct_ans - acc_prox_gd_ans) < 1e-5

