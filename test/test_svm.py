import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.random as npr
import numpy.linalg as npla

import gd

def test_linear_svm():
    n = 10
    p = 5
    npr.seed(124)
    y = npr.binomial(1, 0.5, size=n) * 2 - 1.0
    X = npr.normal(size=(n, p))
    lam = 2.0
    correct_ans = np.array([-0.29246741, 0.17679542, -0.0362559, 0.58550816,
        1.12193581])
    proj_gd_ans = gd.model.linear_svm(y, X, lam=lam,
            max_iter=10000, tol=1e-9, algo="proj_gd")

    assert npla.norm(correct_ans - proj_gd_ans) < 1e-5

    n = 10
    p = 15
    npr.seed(124)
    y = npr.binomial(1, 0.5, size=n) * 2 - 1.0
    X = npr.normal(size=(n, p))
    lam = 2.0
    correct_ans = np.array([-0.20254232, 0.02888906, -0.06402393, -0.04474011,
        0.09855651, -0.12170191, 0.06343478, -0.17205232, -0.04117038,
        -0.1259854, 0.30898503, 0.01598941, 0.09915604, 0.09357455,
        -0.48834549])
    proj_gd_ans = gd.model.linear_svm(y, X, lam=lam,
            max_iter=10000, tol=1e-9, algo="proj_gd")

    assert npla.norm(correct_ans - proj_gd_ans) < 1e-5

