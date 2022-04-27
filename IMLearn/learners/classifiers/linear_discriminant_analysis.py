from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


def mult_mat_by_rows(vector, matrix):
    return (matrix.T * vector).T


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None
        self._counted_classes = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = np.shape(X)
        self.classes_, self._counted_classes = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)

        self.mu_ = np.ndarray((n_classes, n_features))
        for i, k in enumerate(self.classes_):
            self.mu_[i] = np.sum(X[y == k], axis=0)
        self.mu_ = mult_mat_by_rows(1 / self._counted_classes, self.mu_)

        self.cov_ = np.ndarray((n_features, n_features))
        for i, k in enumerate(self.classes_):
            x_sub_mu = X[y == k] - self.mu_[i]
            self.cov_ += x_sub_mu.T @ x_sub_mu
        self.cov_ = (1 / (n_samples - n_classes)) * self.cov_

        self._cov_inv = inv(self.cov_)
        self.pi_ = self._counted_classes / n_samples

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
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples, n_features = np.shape(X)
        n_classes = len(self.classes_)
        likelihoods = np.ndarray((n_samples, n_classes))
        for i, k in enumerate(self.classes_):
            log_pi_k = np.log(self.pi_[i])
            mu_k = 0.5 * (self.mu_[i] @ self._cov_inv @ self.mu_[i])
            for j in range(n_samples):
                likelihoods[j][i] = log_pi_k + X[j] @ self._cov_inv @ self.mu_[i] - mu_k
        return likelihoods

        # sum_on_k = 0
        # for i, k in enumerate(self.classes_):
        #     sum_on_k += self._counted_classes[i] * np.log(self.pi_[i])
        #     sum_on_m = 0
        #     for j in range(len(self.classes_)):
        #         x_sub_mu = X[j] - self.mu_[self.classes_ == y[j]]
        #         sum_on_m += x_sub_mu @ self._cov_inv @ x_sub_mu.T
        #     sum_on_k -= 0.5 * ()
        #
        # part_2 = 0.5 * n_samples * n_features * np.log(2 * np.pi)
        # part_3 = 0.5 * n_samples * np.log(det(self.cov_))
        # return sum_on_k - part_2 - part_3

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
