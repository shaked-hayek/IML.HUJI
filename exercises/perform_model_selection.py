from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


TRAIN_PROPORTION = 2 / 3
K_MIN = 0
K_MAM = 11


def split_and_get_array(X, y, train_proportion):
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.DataFrame(y),
                                                        train_proportion=train_proportion)
    n_samples_train = train_X.shape[0]
    n_samples_test = test_X.shape[0]
    train_X = train_X.to_numpy().reshape((n_samples_train,))
    train_y = train_y.to_numpy().reshape((n_samples_train,))
    test_X = test_X.to_numpy().reshape((n_samples_test,))
    test_y = test_y.to_numpy().reshape((n_samples_test,))
    return train_X, train_y, test_X, test_y


def x_func(x):
    return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    y = x_func(X)
    y_noise = y + np.random.normal(0, scale=noise, size=len(y))
    train_X, train_y, test_X, test_y = split_and_get_array(X, y_noise, train_proportion=TRAIN_PROPORTION)

    go.Figure([
            go.Scatter(x=X, y=y, mode="markers+lines", name="Real Points",
                       marker=dict(color="black", opacity=.9)),
            go.Scatter(x=train_X, y=train_y, mode="markers+lines", name="Train Points",
                       marker=dict(color="red", opacity=.9)),
            go.Scatter(x=test_X, y=test_y, mode="markers+lines", name="Test Points",
                       marker=dict(color="blue", opacity=.9))
    ], layout=go.Layout(
            title={"text": "True, Train and Test Data for {0} samples and noise={1}".format(n_samples, noise)},
            xaxis={"title": r"$x$"},
            yaxis={"title": r"$y$"})).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    avg_scores = []
    for k in range(K_MIN, K_MAM):
        poly_fit = PolynomialFitting(k)
        avg_scores.append(cross_validate(poly_fit, train_X, train_y, mean_square_error, cv=5))
    avg_scores_nd = np.array(avg_scores)

    go.Figure([
        go.Scatter(x=list(range(K_MIN, K_MAM)), y=avg_scores_nd[:, 0], mode="markers+lines", name="Average Train Scores",
                   marker=dict(color="black", opacity=.7)),
        go.Scatter(x=list(range(K_MIN, K_MAM)), y=avg_scores_nd[:, 1], mode="markers+lines", name="Average Validation Scores",
                   marker=dict(color="red", opacity=.7)),
    ], layout=go.Layout(
        title={"text": "Average Train and Validation Scores for {0} samples and noise={1}".format(n_samples, noise)},
        xaxis={"title": "k"},
        yaxis={"title": r"score"})).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    avg_validation_scores = avg_scores_nd[:, 1]
    best_k = int(np.argmin(avg_validation_scores))
    print("The k that minimizes the validation score: {0}".format(best_k))

    best_poly_fit = PolynomialFitting(best_k)
    best_poly_fit.fit(train_X, train_y)
    pred_y = best_poly_fit.predict(test_X)
    print("The test error of the fitted model: {0}".format(mean_square_error(test_y, pred_y)))
    print("The validation error previously achieved with this k: {0}".format(avg_validation_scores[best_k]))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
    # select_regularization_parameter()
