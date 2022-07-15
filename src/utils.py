import yaml
import numpy as np


def get_configs():
    try:
        with open('../config.yaml') as c:
            data = yaml.load(c, Loader=yaml.FullLoader)
    except FileNotFoundError:
        with open('./config.yaml') as c:
            data = yaml.load(c, Loader=yaml.FullLoader)
    return data


def linear_fit(X, y=None):
    if y is None:
        assert X.shape[1] == 2, f'If y is not provided, X.shape[1] == 2. Currently: {X.shape[1]}'
        y = X[:, 1]
        X = X[:, 0]

    try:
        X.shape[1] == 1
    except IndexError:
        X = X.squeeze()

    fit = np.polyfit(X, y, 1)
    x_line = X
    y_line = np.polyval(fit, x_line)

    return x_line, y_line, fit


def check_input(input, options):
    input = input.lower()
    assert input in options, f"input must be in {options}, received {input}"
