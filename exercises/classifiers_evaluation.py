import pandas as pd

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics import accuracy
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be a ndarray of shape (n_samples, 3) where the first 2
    columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load("..\\datasets\\" + filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def get_loss_callback(fit: Perceptron, x_c: np.ndarray, y_c: int):
            losses.append(fit._loss(X, y))

        perceptron = Perceptron(callback=get_loss_callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        px.scatter(
            x=range(len(losses)),
            y=losses,
            labels={"x": "Iteration num", "y": "Loss"},
            title="{0} perceptron training loss values over iterations".format(n)
        ).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else \
        (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and
    gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_predicted_y = lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot
        # titles should specify algorithm and accuracy
        # Create subplots
        lda_accuracy = accuracy(y, lda_predicted_y)
        gaussian_accuracy = 0  # TODO

        symbols = np.array(["circle", "x", "diamond"])
        fig = make_subplots(cols=2, subplot_titles=[
            "Gaussian Naive Bayes predictions - accuracy = {0}".format(gaussian_accuracy),
            "LDA predictions - accuracy = {0}".format(lda_accuracy)
        ])

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=lda_predicted_y.astype(int),
                                             symbol=symbols[y.astype(int)])), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers", showlegend=False,
                                 marker=dict(size=8, color="grey", symbol="x")), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.mu_)):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

        fig.update_layout(title_text="Subplots using {0} dataset".format(f))
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
