import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.random as npr
import numpy.linalg as npla

import gd

def test_linear_L_inf():
    n = 10
    p = 5
    npr.seed(124)
    y = npr.normal(size=n)
    X = npr.normal(size=(n, p))
    lam = 1.0
    correct_ans = np.array([0.18506951, 0.26056539, -0.35884958, 0.23780848,
        -0.35884958])
    prox_gd_ans = gd.model.linear_L_inf(y, X, lam=lam,
            max_iter=2000, tol=1e-9, algo="prox_gd")
    acc_prox_gd_ans = gd.model.linear_L_inf(y, X, lam=lam,
            max_iter=2000, tol=1e-9, algo="acc_prox_gd")

    assert npla.norm(correct_ans - prox_gd_ans) < 1e-5
    assert npla.norm(correct_ans - acc_prox_gd_ans) < 1e-5

    n = 10
    p = 15
    npr.seed(124)
    y = npr.normal(size=n)
    X = npr.normal(size=(n, p))
    lam = 1.0
    correct_ans = np.array([-0.61971445, -0.61971445, 0.40509739, 0.61971445,
        0.26055439, 0.61971445, 0.20940241, -0.39614774, 0.02688884,
        0.61971445, 0.38020918, -0.23735116, -0.61971445, -0.61971445,
        0.35461306])
    prox_gd_ans = gd.model.linear_L_inf(y, X, lam=lam,
            max_iter=2000, tol=1e-9, algo="prox_gd")
    acc_prox_gd_ans = gd.model.linear_L_inf(y, X, lam=lam,
            max_iter=2000, tol=1e-9, algo="acc_prox_gd")

    assert npla.norm(correct_ans - prox_gd_ans) < 1e-5
    assert npla.norm(correct_ans - acc_prox_gd_ans) < 1e-5

