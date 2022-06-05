from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def split_xy(s):
    return s[:, :-1], s[:, -1]


def split_k_folds(X: np.ndarray, y: np.ndarray, k: int):
    S = np.c_[X, y]
    return np.array_split(S, k, axis=0)


def get_train_test(folds, i):
    c_folds = deepcopy(folds)
    test_X, test_y = split_xy(c_folds.pop(i))
    train_X, train_y = split_xy(np.concatenate(c_folds))
    return train_X.reshape((train_X.shape[0],)), train_y, test_X.reshape((test_X.shape[0],)), test_y


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    folds = split_k_folds(X, y, cv)
    train_scores = []
    validation_scores = []
    for i in range(cv):
        train_X, train_y, test_X, test_y = get_train_test(folds, i)
        estimator.fit(train_X, train_y)
        pred_y = estimator.predict(test_X)
        train_pred_y = estimator.predict(train_X)
        train_scores.append(scoring(train_y, train_pred_y))
        validation_scores.append(scoring(test_y, pred_y))
    return np.average(train_scores), np.average(validation_scores)

