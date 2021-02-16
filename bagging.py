#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-31
# @Contact    : qichun.tang@bupt.edu.cn
"""Bagging classifier trained on balanced bootstrap samples."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT
import itertools
import numbers
from warnings import warn

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble._bagging import MAX_INT, _generate_bagging_indices
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.validation import _check_sample_weight


# from ..pipeline import Pipeline
# from ..under_sampling import RandomUnderSampler
# from ..under_sampling.base import BaseUnderSampler
# from ..utils import Substitution, check_target_type
# from ..utils._docstring import _n_jobs_docstring
# from ..utils._docstring import _random_state_docstring


def _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                               seeds, total_n_estimators, verbose):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    # fixme 自己改的, 防止 'fit_params' 导致检测失败
    support_sample_weight = sample_weight is not None
    # support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
    #                                           "sample_weight")
    # if not support_sample_weight and sample_weight is not None:
    #     raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)
        print(estimator)
        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      bootstrap, n_features,
                                                      n_samples, max_features,
                                                      max_samples)
        # 先bagging， 再下采样
        X_ = X[indices]
        y_ = y[indices]
        sampler = RandomUnderSampler(random_state=seeds[i])  # todo 参数化
        X_, y_ = sampler.fit_resample(X_, y_)
        sample_weight_ = None
        if sample_weight is not None:
            sample_weight_ = sample_weight[indices].copy()
            sample_weight_ = sample_weight_[sampler.sample_indices_]
            # todo: 重构样本分布
            # sample_weight_ -= np.min(sample_weight_)
            # sample_weight_ /= np.max(sample_weight_)
            # alpha = 0.1  # todo: 参数化
            # sample_weight_ *= (1 - alpha)
            # sample_weight_ += alpha

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            estimator.fit(X_[:, features], y_, sample_weight=sample_weight_)
        else:
            estimator.fit(X_[:, features], y_)

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


class BalancedBaggingClassifier(BaggingClassifier):
    def __init__(
            self,
            base_estimator=None,
            n_estimators=10,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=False,
            warm_start=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            sampling_strategy="auto",
            replacement=False,
    ):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement

    def fit(self, X, y, sample_weight=None):
        random_state = check_random_state(self.random_state)
        self._max_samples = int(self.max_samples * X.shape[0])
        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(
            X, y, ['csr', 'csc'], dtype=None, force_all_finite=False,
            multi_output=True
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, np.float):
            max_features = self.max_features * self.n_features_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                               **self._parallel_args())(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self
