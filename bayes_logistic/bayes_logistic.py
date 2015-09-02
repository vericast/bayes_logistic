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


from __future__ import absolute_import, print_function
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def logistic_prob(X, w):
    """ MAP (Bayes point) logistic regression probability with overflow prevention via exponent truncation

    Parameters
    ----------
    X : array-like, shape (N, p)
        Feature matrix
    w : array-like, shape (p, )
        Parameter vector

    Returns
    -------

    pr : array-like, shape (N, )
        vector of logistic regression probabilities

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """

    # set a truncation exponent.
    trunc = 8.  # exp(8)/(1+exp(8)) = 0.9997 which is close enough to 1 as to not matter in most cases.

    # calculate argument of logit
    z = np.dot(X, w)

    # truncate to avoid numerical over/underflow
    z = np.clip(z, -trunc, trunc)

    # calculate logitstic probability
    pr = np.exp(z)
    pr = pr / (1. + pr)

    return pr


def f_log_posterior(w, wprior, H, y, X, weights=None):
    """Returns negative log posterior probability.

    Parameters
    ----------
    w : array-like, shape (p, )
        vector of parameters at which the negative log posterior is to be evaluated
    wprior : array-like, shape (p, )
        vector of prior means on the parameters to be fit
    H : array-like, shape (p, p) or (p, )
        Array of prior Hessian (inverse covariance of prior distribution of parameters)
    y : array-like, shape (N, )
        vector of binary ({0,1} responses)
    X : array-like, shape (N, p)
        array of features
    weights : array-like, shape (N, )
        vector of data point weights. Should be within [0,1]

    Returns
    -------
    neg_log_post : float
                negative log posterior probability

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """

    # fill in weights if need be
    if weights is None:
        weights = np.ones(len(np.atleast_1d(y)), )
    if len(np.atleast_1d(weights)) != len(np.atleast_1d(y)):
        raise ValueError(' weight vector must be same length as response vector')

    # calculate negative log posterior

    eps = 1e-6  # this defined to ensure that we never take a log of zero

    mu = logistic_prob(X, w)

    if len(H.shape) == 2:
        neg_log_post = (- (np.dot(y.T, weights * np.log(mu + eps))
                           + np.dot((1. - y).T, weights * np.log(1. - mu + eps)))
                        + 0.5 * np.dot((w - wprior).T, np.dot(H, (w - wprior))))
    elif len(H.shape) == 1:
        neg_log_post = (- (np.dot(y.T, weights * np.log(mu + eps))
                           + np.dot((1. - y).T, weights * np.log(1. - mu + eps)))
                        + 0.5 * np.dot((w - wprior).T, H * (w - wprior)))
    else:
        raise ValueError('Incompatible Hessian')

    return float(neg_log_post)


def g_log_posterior(w, wprior, H, y, X, weights=None):
    """Returns gradient of the negative log posterior probability.

    Parameters
    ----------
    w : array-like, shape (p, )
        parameter vector at which the gradient is to be evaluated
    wprior : array-like, shape (p, )
        array of prior means on the parameters to be fit
    H : array-like, shape (p, p) or (p, )
        array of prior Hessian (inverse covariance of prior distribution of parameters)
    y : array-like, shape (N, )
        array of binary ({0,1} responses)
    X : array-like, shape (N, p)
        array of features
    weights : array-like, shape (N, )
        array of data point weights. Should be within [0,1]

    Returns
    -------
    grad_log_post : array-like, shape (p, )
                gradient of negative log posterior

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """

    # fill in weights if need be
    if weights is None:
        weights = np.ones(len(np.atleast_1d(y)), )
    if len(np.atleast_1d(weights)) != len(np.atleast_1d(y)):
        raise ValueError(' weight vector must be same length as response vector')

    # calculate gradient

    mu_ = logistic_prob(X, w)

    if len(H.shape) == 2:
        grad_log_post = np.dot(X.T, weights * (mu_ - y)) + np.dot(H, (w - wprior))
    elif len(H.shape) == 1:
        grad_log_post = np.dot(X.T, weights * (mu_ - y)) + H * (w - wprior)
    else:
        raise ValueError('Incompatible Hessian')

    return grad_log_post


def g_log_posterior_small(w, wprior, H, y, X, weights=None):
    """Returns normalized (to 1) gradient of the negative log posterior probability.
    This is used for BFGS and L-BFGS-B solvers which tend to not converge unless
    the gradient is normalized.

    Parameters
    ----------
    w : array-like, shape (p, )
        parameter vector at which the gradient is to be evaluated
    wprior : array-like, shape (p, )
        array of prior means on the parameters to be fit
    H : array-like, shape (p, p) or (p, )
        array of prior Hessian (inverse covariance of prior distribution of parameters)
    y : array-like, shape (N, )
        array of binary ({0,1} responses)
    X : array-like, shape (N, p)
        array of features
    weights : array-like, shape (N, )
        array of data point weights. Should be within [0,1]

    Returns
    -------
    grad_log_post : array-like, shape (p, )
                normalized (to 1) gradient of negative log posterior

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """

    # fill in weights if need be
    if weights is None:
        weights = np.ones(len(np.atleast_1d(y)), )
    if len(np.atleast_1d(weights)) != len(np.atleast_1d(y)):
        raise ValueError(' weight vector must be same length as response vector')

    # calculate gradient

    mu = logistic_prob(X, w)

    if len(H.shape) == 2:
        grad_log_post = np.dot(X.T, weights * (mu - y)) + np.dot(H, (w - wprior))
    elif len(H.shape) == 1:
        grad_log_post = np.dot(X.T, weights * (mu - y)) + H * (w - wprior)
    else:
        raise ValueError('Incompatible Hessian')

    # normalize gradient to length 1
    grad_log_post = grad_log_post / np.sqrt(np.sum(grad_log_post * grad_log_post))

    return grad_log_post


def H_log_posterior(w, wprior, H, y, X, weights=None):
    """Returns Hessian (either full or diagonal) of the negative log posterior probability.

    Parameters
    ----------
    w : array-like, shape (p, )
        parameter vector at which the Hessian is to be evaluated
    wprior : array-like, shape (p, )
        array of prior means on the parameters to be fit
    H : array-like, shape (p, p) or (p, )
        array of log prior Hessian (inverse covariance of prior distribution of parameters)
    y : array-like, shape (N, )
        array of binary ({0,1} responses)
    X : array-like, shape (N, p)
        array of features
    weights : array-like, shape (N, )
        array of data point weights. Should be within [0,1]

    Returns
    -------
    H_log_post : array-like, shape like `H`
                Hessian of negative log posterior

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """

    # fill in weights if need be
    if weights is None:
        weights = np.ones(len(np.atleast_1d(y)), )
    if len(np.atleast_1d(weights)) != len(np.atleast_1d(y)):
        raise ValueError(' weight vector must be same length as response vector')

    # calculate log posterior Hessian

    mu = logistic_prob(X, w)

    S = mu * (1. - mu) * weights

    if len(H.shape) == 2:
        H_log_post = np.dot(X.T, X * S[:, np.newaxis]) + H
    elif len(H.shape) == 1:
        H_log_post = np.diag(np.dot(X.T, X * S[:, np.newaxis])) + H
    else:
        raise ValueError('Incompatible Hessian')

    return H_log_post


def HP_log_posterior(w, q, wprior, H, y, X, weights=None):
    """Returns diagonal Hessian of the negative log posterior probability multiplied by an arbitrary vector.
    This is useful for the Newton-CG solver, particularly when we only want to store a diagonal Hessian.

    Parameters
    ----------
    w : array-like, shape (p, )
        parameter vector at which the Hessian is to be evaluated
    q : array-like, shape (p, )
        arbitrary vector to multiply Hessian by
    wprior : array-like, shape (p, )
        array of prior means on the parameters to be fit
    H : array-like, shape (p, )
        array of diagonal log prior Hessian (inverse covariance of prior distribution of parameters)
    y : array-like, shape (N, )
        array of binary ({0,1} responses)
    X : array-like, shape (N, p)
        array of features
    weights : array-like, shape (N, )
        array of data point weights. Should be within [0,1]

    Returns
    -------
    HP : array-like, shape (p, )
        Hessian of log posterior (diagonal approx) multiplied by arbitrary vector

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """

    # fill in weights if need be
    if weights is None:
        weights = np.ones(len(np.atleast_1d(y)), )
    if len(np.atleast_1d(weights)) != len(np.atleast_1d(y)):
        raise ValueError(' weight vector must be same length as response vector')

    HP = H_log_posterior(w, wprior, H, y, X, weights)
    HP = HP * q

    return HP


def fit_bayes_logistic(y, X, wprior, H, weights=None, solver='Newton-CG', bounds=None, maxiter=100):
    """ Bayesian Logistic Regression Solver.  Assumes Laplace (Gaussian) Approximation
    to the posterior of the fitted parameter vector. Uses scipy.optimize.minimize

    Parameters
    ----------
    y : array-like, shape (N, )
        array of binary {0,1} responses
    X : array-like, shape (N, p)
        array of features
    wprior : array-like, shape (p, )
        array of prior means on the parameters to be fit
    H : array-like, shape (p, p) or (p, )
        array of prior Hessian (inverse covariance of prior distribution of parameters)
    weights : array-like, shape (N, )
        array of data point weights. Should be within [0,1]
    solver : string
        scipy optimize solver used.  this should be either 'Newton-CG', 'BFGS' or 'L-BFGS-B'.
        The default is Newton-CG.
    bounds : iterable of length p
        a length p list (or tuple) of tuples each of length 2.
        This is only used if the solver is set to 'L-BFGS-B'. In that case, a tuple
        (lower_bound, upper_bound), both floats, is defined for each parameter.  See the
        scipy.optimize.minimize docs for further information.
    maxiter : int
        maximum number of iterations for scipy.optimize.minimize solver.

    Returns
    -------
    w_fit : array-like, shape (p, )
        posterior parameters (MAP estimate)
    H_fit : array-like, shape like `H`
        posterior Hessian  (Hessian of negative log posterior evaluated at MAP parameters)

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """

    # Check that dimensionality of inputs agrees

    # check X
    if len(X.shape) != 2:
        raise ValueError('X must be a N*p matrix')
    (nX, pX) = X.shape

    # check y
    if len(y.shape) > 1:
        raise ValueError('y must be a vector of shape (p, )')
    if len(np.atleast_1d(y)) != nX:
        raise ValueError('y and X do not have the same number of rows')

    # check wprior
    if len(wprior.shape) > 1:
        raise ValueError('prior should be a vector of shape (p, )')
    if len(np.atleast_1d(wprior)) != pX:
        raise ValueError('prior mean has incompatible length')

    # check H
    if len(H.shape) == 1:
        if np.atleast_1d(H).shape[0] != pX:
            raise ValueError('prior Hessian is diagonal but has incompatible length')
    elif len(H.shape) == 2:
        (h1,h2) = np.atleast_2d(H).shape
        if h1 != h2:
            raise ValueError('prior Hessian must either be a p*p square matrix or a vector or shape (p, ) ')
        if h1 != pX:
            raise ValueError('prior Hessian is square but has incompatible size')

    # fill in weights if need be
    if weights is None:
        weights = np.ones(len(np.atleast_1d(y)), )
    if len(np.atleast_1d(weights)) != len(np.atleast_1d(y)):
        raise ValueError(' weight vector must be same length as response vector')

    # Do the regression

    if solver == 'Newton-CG':

        if len(H.shape) == 2:

            ww = minimize(f_log_posterior, wprior, args=(wprior, H, y, X, weights), jac=g_log_posterior,
                          hess=H_log_posterior, method='Newton-CG', options={'maxiter': maxiter})
            w_fit = ww.x
            H_fit = H_log_posterior(w_fit, wprior, H, y, X, weights)

        elif len(H.shape) == 1:

            ww = minimize(f_log_posterior, wprior, args=(wprior, H, y, X, weights), jac=g_log_posterior,
                          hessp=HP_log_posterior, method='Newton-CG', options={'maxiter': maxiter})
            w_fit = ww.x
            H_fit = H_log_posterior(w_fit, wprior, H, y, X, weights)

        else:
            raise ValueError(' You must either use the full Hessian or its diagonal as a vector')

    elif solver == 'BFGS':
        ww = minimize(f_log_posterior, wprior, args=(wprior, H, y, X, weights), jac=g_log_posterior_small,
                      method='BFGS', options={'maxiter': maxiter})
        w_fit = ww.x
        H_fit = H_log_posterior(w_fit, wprior, H, y, X, weights)

    elif solver == 'L-BFGS-B':
        ww = minimize(f_log_posterior, wprior, args=(wprior, H, y, X, weights), jac=g_log_posterior_small,
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter})
        w_fit = ww.x
        H_fit = H_log_posterior(w_fit, wprior, H, y, X, weights)

    else:
        raise ValueError('Unknown solver specified: "{0}"'.format(solver))

    return w_fit, H_fit


def get_pvalues(w, H):
    """ Calculates p-values on fitted parameters. This can be used for variable selection by,
    for example, discarding every parameter with a p-value less than 0.05 (or some other cutoff)

    Parameters
    ----------
    w : array-like, shape (p, )
        array of posterior means on the fitted parameters
    H : array-like, shape (p, p) or (p, )
        array of log posterior Hessian

    Returns
    -------
    pvals : array-like, shape (p, )
        array of p-values for each of the fitted parameters

    References
    ----------
    Chapter 2 of Pawitan, Y. 'In All Likelihood', Oxford University Press (2013)
    Also see: Gerhard, F. 'Extraction of network topology from multi-electrode recordings: is there
    a small world effect', Frontiers in Computational Neuroscience (2011) for a use case of
    p-value based variable selection.

    """

    # get inverse standard error of each parameter from the square root of the Hessian,
    # which is equal to the Fisher information
    if len(H.shape) == 2:
        inv_std_err = np.sqrt(np.diag(H))
    elif len(H.shape) == 1:
        inv_std_err = np.sqrt(H)
    else:
        raise ValueError("Incompatible Hessian provided")

    # calculate Wald statistic
    z_ = w * inv_std_err

    # get p-value by comparing Wald statistic to cdf of Normal distribution
    pvals = 2. * (1. - norm.cdf(np.abs(z_)))

    return pvals


def bayes_logistic_prob(X, w, H):
    """ Posterior predictive logistic regression probability.  Uses probit approximation
        to the logistic regression sigmoid. Also has overflow prevention via exponent truncation.

    Parameters
    ----------
    X : array-like, shape (N, p)
        array of covariates
    w : array-like, shape (p, )
        array of fitted MAP parameters
    H : array-like, shape (p, p) or (p, )
        array of log posterior Hessian (covariance matrix of fitted MAP parameters)

    Returns
    -------
    pr : array-like, shape (N, )
        moderated (by full distribution) logistic probability

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """

    # set a truncation exponent
    trunc = 8.  # exp(8)/(1+exp(8)) = 0.9997 which is close enough to 1 as to not matter in most cases.

    # unmoderated argument of exponent
    z_a = np.dot(X, w)

    # find  the moderation
    if len(H.shape) == 2:
        H_inv_ = np.linalg.inv(H)
        sig2_a = np.sum(X * np.dot(H_inv_, X.T).T, axis=1)
    elif len(H.shape) == 1:
        H_inv_ = 1. / H
        sig2_a = np.sum(X * (H_inv_ * X), axis=1)
    else:
        raise ValueError(' You must either use the full Hessian or its diagonal as a vector')

    # get the moderation factor. Implicit in here is approximating the logistic sigmoid with
    # a probit by setting the probit and sigmoid slopes to be equal at the origin. This is where
    # the factor of pi/8 comes from.
    kappa_sig2_a = 1. / np.sqrt(1. + 0.125 * np.pi * sig2_a)

    # calculate the moderated argument of the logit
    z = z_a * kappa_sig2_a

    # do a truncation to prevent exp overflow
    z = np.clip(z, -trunc, trunc)

    # get the moderated logistic probability
    pr = np.exp(z)
    pr = pr / (1. + pr)

    return pr
