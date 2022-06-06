from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics.loss_functions import mean_square_error


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """


        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _add_intercept(self, X: np.ndarray):
        if self.include_intercept_:
            return np.c_[np.ones(X.shape[0]), X]
        return X

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        X = self._add_intercept(X)
        # U, S, V = np.linalg.svd(X, full_matrices=False)
        # S_trans = np.transpose(S)
        # S_lambda = np.linalg.inv((S_trans @ S) + self.lam_ * np.identity(S.shape[0])) @ S_trans
        # self.coefs_ = V @ S_lambda @ np.transpose(U)

        d = X.shape[1]
        lam_identity = np.sqrt(self.lam_) * np.identity(d)
        X_lam = np.concatenate((X, lam_identity), axis=0)
        y_lam = np.concatenate((y, np.zeros(d)), axis=0)
        self.coefs_ = np.matmul(np.linalg.pinv(X_lam), y_lam)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        X = self._add_intercept(X)
        return np.matmul(X, self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_pred = self._predict(X)
        return mean_square_error(y, y_pred)
