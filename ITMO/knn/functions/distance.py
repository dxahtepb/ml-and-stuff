import functools


def minkowski(a, b, p):
    mink_sum = 0
    for a_x, b_x in zip(a.coords, b.coords):
        mink_sum += abs(a_x - b_x)**p
    return mink_sum**(1 / p)


def chebyshev(a, b):
    return max([
        abs(a_x - b_x)
        for a_x, b_x in zip(a.coords, b.coords)
    ])


euclidean = functools.partial(minkowski, p=2)
manhattan = functools.partial(minkowski, p=1)
