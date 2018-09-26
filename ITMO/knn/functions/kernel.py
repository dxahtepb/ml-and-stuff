import math


def uniform(u):
    return 0.5


def gaussian(u):
    return math.e**(-u*u / 2) / (2 * math.pi)**(1 / 2)


def sigmoid(u):
    return 2 / (math.pi * (math.e**u + math.e**(-u)))


def triangular(u):
    return 1 - abs(u)


def quartic(u):
    return 15 * ((1 - u*u)*(1 - u*u)) / 16


def epanechnikov(u):
    return 3 * (1 - u*u) / 4


def logistic(u):
    return 1 / (math.e**u + 2 + math.e**(-u))


def cosine(u):
    return math.pi/4 * math.cos(math.pi/2 * u)
