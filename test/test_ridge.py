import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.random as npr
import numpy.linalg as npla

import gd

def test_linear_ridge():
    n = 10
    p = 5
    npr.seed(124)
    y = npr.normal(size=n)
    X = npr.normal(size=(n, p))
    lam = 1.0
    correct_ans = np.array([0.17665115, 0.24005085, -0.46077346, 0.22848636,
        -0.340608])
    gd_ans = gd.model.linear_ridge(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="gd")
    acc_gd_ans = gd.model.linear_ridge(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="acc_gd")
    prox_gd_ans = gd.model.linear_ridge(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="prox_gd")
    acc_prox_gd_ans = gd.model.linear_ridge(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="acc_prox_gd")

    assert npla.norm(correct_ans - gd_ans) < 1e-5
    assert npla.norm(correct_ans - acc_gd_ans) < 1e-5
    assert npla.norm(correct_ans - prox_gd_ans) < 1e-5
    assert npla.norm(correct_ans - acc_prox_gd_ans) < 1e-5

    n = 10
    p = 15
    npr.seed(124)
    y = npr.normal(size=n)
    X = npr.normal(size=(n, p))
    lam = 1.0
    correct_ans = np.array([-0.371691, -0.65851918, 0.11325059, 0.34433978,
        0.29712003, 0.26002903, -0.06493076, -0.20457246, -0.0474628,
        0.43189872, 0.10062145, 0.05488053, -0.21244535, -0.23388861,
        0.36954105])
    gd_ans = gd.model.linear_ridge(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="gd")
    acc_gd_ans = gd.model.linear_ridge(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="acc_gd")
    prox_gd_ans = gd.model.linear_ridge(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="prox_gd")
    acc_prox_gd_ans = gd.model.linear_ridge(y, X, lam=lam,
            max_iter=2000, tol=1e-8, algo="acc_prox_gd")

    assert npla.norm(correct_ans - gd_ans) < 1e-4
    assert npla.norm(correct_ans - acc_gd_ans) < 1e-4
    assert npla.norm(correct_ans - prox_gd_ans) < 1e-4
    assert npla.norm(correct_ans - acc_prox_gd_ans) < 1e-4

