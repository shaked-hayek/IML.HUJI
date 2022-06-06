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


def reshape_array(arr):
    return arr.to_numpy().reshape((arr.shape[0],))


def split_and_get_array(X, y, train_proportion):
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y),
                                                        train_proportion=train_proportion)
    return reshape_array(train_X), reshape_array(train_y), reshape_array(test_X), reshape_array(test_y)


def x_func(x):
    return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)


def plot_k_folds_result(avg_scores, x_range, plot_title, x_title):
    go.Figure([
        go.Scatter(x=x_range, y=avg_scores[:, 0], mode="markers+lines",
                   name="Average Train Scores", marker=dict(color="black", opacity=.7)),
        go.Scatter(x=x_range, y=avg_scores[:, 1], mode="markers+lines",
                   name="Average Validation Scores", marker=dict(color="red", opacity=.7)),
    ], layout=go.Layout(
        title={"text": plot_title},
        xaxis={"title": x_title},
        yaxis={"title": r"score"})).show()


def report_best_param(avg_scores, param_name, model=""):
    avg_validation_scores = avg_scores[:, 1]
    best_param = int(np.argmin(avg_validation_scores))
    print("The {0} that minimizes the validation score{1}: {2}".format(param_name, model, best_param))
    return best_param


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
    y_noise = y + np.random.normal(0, scale=noise, size=n_samples)
    train_X, train_y, test_X, test_y = split_and_get_array(X, y_noise, train_proportion=TRAIN_PROPORTION)

    go.Figure([
            go.Scatter(x=X, y=y, mode="markers", name="Real Points",
                       marker=dict(color="black", opacity=.9)),
            go.Scatter(x=train_X, y=train_y, mode="markers", name="Train Points",
                       marker=dict(color="red", opacity=.9)),
            go.Scatter(x=test_X, y=test_y, mode="markers", name="Test Points",
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

    plot_k_folds_result(avg_scores_nd, list(range(K_MIN, K_MAM)), x_title="k",
                        plot_title="Average Train and Validation Scores for {0} samples and noise={1}".format(n_samples, noise))

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = report_best_param(avg_scores_nd, "k")

    best_poly_fit = PolynomialFitting(best_k)
    best_poly_fit.fit(train_X, train_y)
    pred_y = best_poly_fit.predict(test_X)
    print("The test error of the fitted model: {0}".format(mean_square_error(test_y, pred_y)))
    print("The validation error previously achieved with this k: {0}".format(avg_scores[:, 1][best_k]))


def select_regularization_parameter(n_train: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_train: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    n_samples = X.shape[0]
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=n_train/n_samples)


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    reg_param_options = np.linspace(0, 2, n_evaluations)
    avg_scores_lasso = []
    avg_scores_ridge = []
    for reg in reg_param_options:
        lasso_model = Lasso(alpha=reg)
        # ridge_model = RidgeRegression(lam=reg)
        avg_scores_lasso.append(cross_validate(lasso_model, train_X.to_numpy(), train_y.to_numpy(),
                                               mean_square_error, cv=5))
        # avg_scores_ridge.append(cross_validate(ridge_model, train_X.to_numpy(), train_y.to_numpy(),
        #                                        mean_square_error, cv=5))

    avg_scores_lasso = np.array(avg_scores_lasso)
    # avg_scores_ridge = np.array(avg_scores_ridge)

    plot_k_folds_result(avg_scores_lasso, reg_param_options, x_title="regression param",
                        plot_title="Average Train and Validation Scores for {0} model".format("Lasso"))
    # plot_k_folds_result(avg_scores_ridge, reg_param_options, x_title="regression param",
    #                     plot_title="Average Train and Validation Scores for {0} model".format("Ridge"))

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_reg_param_lasso = report_best_param(avg_scores_lasso, "regression param", " of Lasso")
    # best_reg_param_lasso = report_best_param(avg_scores_ridge, "regression param", " of Ridge")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
    # select_regularization_parameter()
