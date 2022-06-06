from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def get_train_test(folds_X, folds_y, i):
    return np.concatenate(np.delete(folds_X, i, 0), axis=0), \
           np.concatenate(np.delete(folds_y, i, 0), axis=0), \
           folds_X[i], folds_y[i]


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
    folds_X = np.array_split(X, cv, axis=0)
    folds_y = np.array_split(y, cv, axis=0)
    train_scores = []
    validation_scores = []
    for i in range(cv):
        train_X, train_y, test_X, test_y = get_train_test(folds_X, folds_y, i)
        estimator.fit(train_X, train_y)
        train_pred_y = estimator.predict(train_X)
        pred_y = estimator.predict(test_X)
        train_scores.append(scoring(train_y, train_pred_y))
        validation_scores.append(scoring(test_y, pred_y))
    return np.average(train_scores), np.average(validation_scores)

