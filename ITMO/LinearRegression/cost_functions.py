from ITMO.LinearRegression.dataset import Point


def mse(expected, predicted):
    if isinstance(expected[0], Point):
        squares_sum = sum([
            (exp.target - pred.target)**2 for exp, pred in zip(expected, predicted)
        ])
    else:
        squares_sum = sum([
            (exp - pred) ** 2 for exp, pred in zip(expected, predicted)
        ])
    return squares_sum / len(expected)


def rmse(expected, predicted):
    return mse(expected, predicted) ** (1/2)
