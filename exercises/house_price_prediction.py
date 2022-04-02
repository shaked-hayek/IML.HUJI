from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from os import path
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

DATASET_FILE = "../datasets/house_prices.csv"
LABEL_NAME = "price"
SHOULD_DROP = ["id", "date"]
DUMMIES = ["zipcode"]


def replace_yr_renovated(row):
    if row["yr_renovated"] == 0:
        row["yr_renovated"] = row["yr_built"]
    return row


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df.fillna(0, inplace=True)

    # Remove rows with id = 0 & negative prices
    df.drop(df[(df.id == 0) | (df.price <= 0)].index, inplace=True)

    # Change yr_renovated
    df = df.apply(replace_yr_renovated, axis=1)

    y = df[LABEL_NAME]
    df = pd.get_dummies(data=df, columns=DUMMIES, prefix="zipcode", dummy_na=True)
    df.drop(columns=[LABEL_NAME] + SHOULD_DROP, inplace=True)

    return df, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X.columns:
        cov = np.cov(X[col], y)[0][1]
        pearson_correlation = cov / (np.std(X[col]) * np.std(y))

        with open(path.join(output_path, "pcorr_{0}.png".format(X[col].name)), "wb") as f:
            img_data = pio.to_image(go.Figure([go.Scatter(x=X[col], y=y, mode='markers')],
                                    layout=go.Layout(
                                        title={"text": "Feature {0} - Pearson Correlation is {1}".format(
                                            X[col].name, pearson_correlation)},
                                        xaxis={"title": "Feature Value"},
                                        yaxis={"title": "Response"})),
                                    format="png")
            f.write(img_data)


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(DATASET_FILE)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "plots")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    #raise NotImplementedError()
