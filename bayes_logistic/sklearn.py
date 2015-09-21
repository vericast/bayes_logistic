# Copyright (c) 2015 MaxPoint Interactive, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#    disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import
from sklearn.utils.validation import NotFittedError

from .bayes_logistic import fit_bayes_logistic as _fit, bayes_logistic_prob as _predict
import six
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression


class BayesLogisticBase(six.with_metaclass(ABCMeta, BaseEstimator), ClassifierMixin):
    """Bayesian Logistic Regression Solver.

    Assumes Laplace (Gaussian) Approximation to the posterior of the fitted parameter vector.
    Uses scipy.optimize.minimize.

    Initial priors for the coefficients and the hessian need not be provided.  If not provided these will be initialized
    to an uninformative initial solution equivalent to an L2 penalty.

    Parameters
    ----------
    coef : array-like, shape (n_features, ), optional
        array of prior means on the parameters to be fit
    H : array-like, shape (n_features, n_features) or (n_features, ), optional
        array of prior Hessian (inverse covariance of prior distribution of parameters)
    solver : string
        scipy optimize solver used.  this should be either 'Newton-CG', 'BFGS' or 'L-BFGS-B'.
        The default is Newton-CG.
    bounds : iterable of length p
        A length p list (or tuple) of tuples each of length 2.
        This is only used if the solver is set to 'L-BFGS-B'. In that case, a tuple
        (lower_bound, upper_bound), both floats, is defined for each parameter.  See the
        scipy.optimize.minimize docs for further information.
    maxiter : int
        Maximum number of iterations for scipy.optimize.minimize solver.
    """

    def __init__(self, H=None, coef=None, solver='Newton-CG',  bounds=None, maxiter=100):
        self.maxiter = maxiter
        self.bounds = bounds
        self.H_ = H
        self.coef_ = coef
        self.solver = solver

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : {array-like, None}, shape (n_samples,)
            Optional weight vector to weight each observation by.  Weights expected to be in [0,1].

        Returns
        -------
        self : object
            Returns self.
        """

        self.partial_fit(X, y, sample_weight)
        return self

    def partial_fit(self, X, y, sample_weight=None):
        """Update the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : {array-like, None}, shape (n_samples,)
            Optional weight vector to weight each observation by.  Weights expected to be in [0,1].

        Returns
        -------
        self : object
            Returns self.
        """
        self._ensure_valid_wh(X, y)
        self.coef_, self.H_ = _fit(y=y, X=X, wprior=self.coef_, H=self.H_, solver=self.solver, maxiter=self.maxiter,
                                   weights=sample_weight, bounds=self.bounds)
        return self

    def predict(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, ]
            Returns the probability of the sample for the model
        """
        if not hasattr(self, 'coef_') or self.coef_ is None:
            raise NotFittedError("This %(name)s instance is not fitted"
                                 "yet" % {'name': type(self).__name__})
        return _predict(X, self.coef_, self.H_)

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, ]
            Returns the probability of the sample for the model
        """
        if not hasattr(self, 'coef_') or self.coef_ is None:
            raise NotFittedError("This %(name)s instance is not fitted"
                                 "yet" % {'name': type(self).__name__})
        return _predict(X, self.coef_, self.H_)

    def _ensure_valid_wh(self, X, y):
        """Ensures that there are valid values for the coefficients and the hessian.

        If not initialized this sets the coefficients and hessian to be equivalent to the L2 penalty
        """
        dim = X.shape[1]
        if self.H_ is None:
            self.H_ = np.diag(np.ones(dim), dim) * 0.001
        if self.coef_ is None:
            self.coef_ = np.zeros(dim)


class BayesLogisticClassifier(BayesLogisticBase, ClassifierMixin):
    pass


class BayesLogisticRegressor(BayesLogisticBase, RegressorMixin):
    pass

__all__ = ["BayesLogisticClassifier", "BayesLogisticRegressor"]
