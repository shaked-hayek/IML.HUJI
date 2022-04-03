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
    fig = px.bar(df_month_grouped, y=TEMP, labels={"Temp": "STD of temp"},
                 title="Each month the standard deviation of the daily temperatures").show()

    # Question 3 - Exploring differences between countries

    # Question 4 - Fitting model for different values of `k`

    # Question 5 - Evaluating fitted model on different countries
