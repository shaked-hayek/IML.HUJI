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


def load_data(filename: str, process=True):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    process: Default True - should process the data

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df.fillna(0, inplace=True)

    if process:
        # Remove rows with id = 0 & negative prices & more than 15 bedrooms
        df.drop(df[(df.id == 0) | (df.price <= 0) | (df.bedrooms > 15)].index, inplace=True)

        # Change yr_renovated
        df = df.apply(replace_yr_renovated, axis=1)

        # Turn Zipcode to dummies
        df = pd.get_dummies(data=df, columns=DUMMIES, prefix="zipcode", dummy_na=True)

    y = df[LABEL_NAME]
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
    std_y = np.std(y)
    for col in X.columns:
        cov = np.cov(X[col], y)[0][1]
        std_x = np.std(X[col])
        if std_x == 0:
            pearson_correlation = 0
        else:
            pearson_correlation = cov / (np.std(X[col]) * std_y)

        with open(path.join(output_path, "pcorr_{0}.png".format(X[col].name)), "wb") as f:
            img_data = pio.to_image(go.Figure([go.Scatter(x=X[col], y=y, mode='markers')],
                                    layout=go.Layout(
                                        title={"text": "Feature {0} - Pearson Correlation is {1}".format(
                                            X[col].name, pearson_correlation)},
                                        xaxis={"title": "Feature Value"},
                                        yaxis={"title": "Response"})),
                                    format="png")
            f.write(img_data)


def fit_model_over_percentages():
    model = LinearRegression()
    percents = np.linspace(10, 100, num=91)
    train_X[LABEL_NAME] = train_y

    mean_losses = []
    std_loss_up = []
    std_loss_down = []
    for per in percents:
        p_loss = np.array([])
        for i in range(10):
            sample_data = train_X.sample(frac=(per / 100))
            sample_y = sample_data[LABEL_NAME]
            sample_X = sample_data.drop(columns=[LABEL_NAME])

            model.fit(sample_X.to_numpy(), sample_y)
            p_loss = np.append(p_loss, model.loss(test_X.to_numpy(), test_y.to_numpy()))
        mean_loss = p_loss.mean()
        std_loss = np.std(p_loss)
        mean_losses.append(mean_loss)
        std_loss_up.append(mean_loss + (2 * std_loss))
        std_loss_down.append(mean_loss - (2 * std_loss))

    go.Figure([go.Scatter(name="MSE Loss", x=percents, y=mean_losses, mode='markers+lines'),
               go.Scatter(x=percents, y=std_loss_up, fill=None, mode="lines", line=dict(color="lightgrey"),
                          showlegend=False),
               go.Scatter(x=percents, y=std_loss_down, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                          showlegend=False)
               ],
              layout=go.Layout(title=r"$\text{Linear regression model over percentages of the training set - "
                                     r"Mean Loss calculation}$",
                               xaxis={"title": "Percent of training set"},
                               yaxis={"title": "Mean Loss"})).show()


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
    fit_model_over_percentages()

