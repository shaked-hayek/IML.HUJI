from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

MEAN = 10
VAR = 1
SAMPLE_NUM = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(loc=MEAN, scale=VAR, size=SAMPLE_NUM)
    estimator = UnivariateGaussian()
    estimator.fit(samples)
    print("({0}, {1})".format(estimator.mu_, estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    ms = np.linspace(10, 1000, 100).astype(int)
    for m in ms:
        #sample_group = np.random.choice(samples, size=m)
        sample_group = samples[:(m + 1)]
        diff = abs(MEAN - np.mean(sample_group))
        estimated_mean.append(diff)
    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines')],
              layout=go.Layout(title=r"$\text{Absolute Distance Between The Estimated And True Value Of The "
                                     r"Expectation As A Function Of The Sample Size}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$|\hat\mu - \mu|$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    samples_pdf = estimator.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=samples_pdf, mode='markers')],
              layout=go.Layout(title=r"$\text{PDF of 1000 N(10,1) samples}$",
                               xaxis_title=r"$\text{sample value}$",
                               yaxis_title=r"$\text{Density}$",
                               height=300)).show()

    # Temp test
    #print(UnivariateGaussian.log_likelihood(10, 1, samples))


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    #test_multivariate_gaussian()
