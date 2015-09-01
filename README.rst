=========================
Bayes Logistic Regression
=========================

.. image:: https://img.shields.io/travis/MaxPoint/bayes_logistic.svg
        :target: https://travis-ci.org/MaxPoint/bayes_logistic

.. image:: https://img.shields.io/pypi/v/bayes_logistic.svg
        :target: https://pypi.python.org/pypi/bayes_logistic


This package will fit Bayesian logistic regression models with arbitrary
prior means and covariance matrices, although we work with the inverse covariance matrix which is the log-likelihood
Hessian.

Either the full Hessian or a diagonal approximation may be used.

Individual data points may be weighted in an arbitrary manner.  

Finally, p-values on each fitted parameter may be calculated and this can be used
for variable selection of sparse models.

* Free software: BSD license
* Documentation: https://bayes_logistic.readthedocs.org.

Demo
----

`Example Notebook`_

.. _Example Notebook: notebooks/bayeslogistic_demo.ipynb

