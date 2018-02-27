import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.random as npr
import numpy.linalg as npla

import gd

def test_linear_slope():
    n = 10
    p = 5
    npr.seed(124)
    y = npr.normal(size=n)
    X = npr.normal(size=(n, p))
    lam = 1.0
    theta = np.linspace(0.02, 0.1, p)[::-1]
    correct_ans = np.array([0.22963392, 0.27827052, -0.56127911, 0.23676543,
        -0.35330124])
    prox_gd_ans = gd.model.linear_slope(y, X, lam=lam, theta=theta,
            max_iter=10000, tol=1e-9, algo="prox_gd")
    acc_prox_gd_ans = gd.model.linear_slope(y, X, lam=lam, theta=theta,
            max_iter=10000, tol=1e-9, algo="acc_prox_gd")

    assert npla.norm(correct_ans - prox_gd_ans) < 1e-5
    assert npla.norm(correct_ans - acc_prox_gd_ans) < 1e-5

    n = 10
    p = 15
    npr.seed(124)
    y = npr.normal(size=n)
    X = npr.normal(size=(n, p))
    lam = 1.0
    theta = np.linspace(0.02, 0.1, p)[::-1]
    correct_ans = np.array([-0.7616362, -1.13867604, 0.0, 0.80449455,
        0.40278817, 0.40278817, 0.02155191, -0.28373785, 0.0, 0.40278817,
        0.12195459, 0.0, 0.0, -0.04964992, 0.56319276])
    prox_gd_ans = gd.model.linear_slope(y, X, lam=lam, theta=theta,
            max_iter=2000, tol=1e-8, algo="prox_gd")
    acc_prox_gd_ans = gd.model.linear_slope(y, X, lam=lam, theta=theta,
            max_iter=2000, tol=1e-8, algo="acc_prox_gd")

    assert npla.norm(correct_ans - prox_gd_ans) < 1e-5
    assert npla.norm(correct_ans - acc_prox_gd_ans) < 1e-5

