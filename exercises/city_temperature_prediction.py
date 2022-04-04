import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

DATASET_FILE = "../datasets/City_Temperature.csv"
DAY_OF_YEAR = "DayOfYear"
DATE = "Date"
YEAR = "Year"
MONTH = "Month"
TEMP = "Temp"
COUNTRY = "Country"
ISRAEL = "Israel"


def load_data(filename: str):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix
    """
    df = pd.read_csv(filename, parse_dates=[DATE])

    # Remove rows with temp < -50
    df.drop(df[df.Temp < -50].index, inplace=True)

    # Add day of year column
    df[DAY_OF_YEAR] = df[DATE].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(DATASET_FILE)

    # Question 2 - Exploring data for specific country
    df[YEAR] = df[YEAR].astype(str)
    df_israel = df.loc[df[COUNTRY] == ISRAEL]

    px.scatter(df_israel, x=DAY_OF_YEAR, y=TEMP, color=YEAR,
               title="Average daily temperature change as a function of the day in year").show()

    df_month_grouped = df_israel.groupby(MONTH).agg(np.std)
    px.bar(df_month_grouped, y=TEMP, labels={"Temp": "STD of temp"},
           title="Each month the standard deviation of the daily temperatures").show()

    # Question 3 - Exploring differences between countries
    df_grouped_3 = df.groupby([MONTH, COUNTRY]).Temp.agg([np.mean, np.std])
    px.line(df_grouped_3, x=df_grouped_3.axes[0].get_level_values(0), y="mean", error_y="std",
            color=df_grouped_3.axes[0].get_level_values(1), labels={"x": "Month", "mean": "Temp mean"},
            title="Average monthly temperature (with STD and country coded)").show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df_israel[DAY_OF_YEAR], df_israel[TEMP])
    loss_l = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = model.loss(test_X.to_numpy(), test_y.to_numpy())
        print("For k={0} loss is {1:.2f}".format(k, loss))
        loss_l.append(loss)
    px.bar(x=list(range(1, 11)), y=loss_l, labels={"x": "k (degree)", "y": "Loss"},
           title="Loss for each value of k (degree)").show()

    # Question 5 - Evaluating fitted model on different countries
    k_chosen = 5
    model = PolynomialFitting(k_chosen)
    model.fit(df_israel[DAY_OF_YEAR], df_israel[TEMP])
    countries = df.Country.unique()
    countries = countries[countries != ISRAEL]
    loss_l = []
    for country in countries:
        df_c = df.loc[df[COUNTRY] == country]
        loss = model.loss(df_c[DAY_OF_YEAR], df_c[TEMP])
        loss_l.append(loss)
    px.bar(x=countries, y=loss_l, labels={"x": "Country", "y": "Loss"},
           title="Loss for countries over model fit to Israel").show()
