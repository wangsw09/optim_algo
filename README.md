# optim_algo
play with new convex optimization algorithms; applied on popular statistics models

* Coded
    * gradient descent (GD);
    * accelerated gradient descent;
    * proximal gradient descent;
    * accelerated proximal gradient descent (FISTA)
    * projected gradient descent
* Applied the above results on the following models (already coded)
    * LASSO (linear)
    * Ridge (linear)
    * svm (linear)
    * L_inf minimization (linear)
    * sorted L1 (SLOPE)
    * nuclear norm minimization
    * positive quadrant constrained ridge regression

* To do:
    * For algo part, will play with several restart regimes;
    * For model, will add generalized linear model (glm) if applicable; also fused
      lasso; and other constrained problems;
    * Then will consider add stochastic optimization algorithms;
