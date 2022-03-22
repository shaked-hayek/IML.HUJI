from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"

MEAN = 10
VAR = 1
SAMPLE_SIZE = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(loc=MEAN, scale=VAR, size=SAMPLE_SIZE)
    estimator = UnivariateGaussian().fit(samples)
    print("({0}, {1})".format(estimator.mu_, estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    ms = np.linspace(10, 1000, 100).astype(int)
    for m in ms:
        sample_group = samples[:m]
        estimator = UnivariateGaussian().fit(sample_group)
        diff = abs(MEAN - estimator.mu_)
        estimated_mean.append(diff)
    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines')],
              layout=go.Layout(title=r"$\text{Absolute Distance Between The Estimated And True Value Of The "
                                     r"Expectation As A Function Of The Sample Size}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$|\hat\mu - \mu|$")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    samples_pdf = estimator.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=samples_pdf, mode='markers')],
              layout=go.Layout(title=r"$\text{PDF of 1000 N(10,1) samples}$",
                               xaxis_title=r"$\text{sample value}$",
                               yaxis_title=r"$\text{Density}$")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean=mu, cov=cov, size=SAMPLE_SIZE)
    estimator = MultivariateGaussian().fit(samples)
    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    results = {}
    matrix = []
    ms = np.linspace(-10, 10, 200)
    for f1 in ms:
        row = []
        for f3 in ms:
            mu = np.array([f1, 0, f3, 0])
            lh = MultivariateGaussian.log_likelihood(mu, cov, samples)
            results[(f1, f3)] = lh
            row.append(lh)
        matrix.append(row)
    px.imshow(matrix, x=ms, y=ms, labels={"x": "f1", "y": "f3"},
              title="Heatmap of log likelihood").show()

    # Question 6 - Maximum likelihood
    print("%.3f, %.3f" % max(results, key=results.get))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
