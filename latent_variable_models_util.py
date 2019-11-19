import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import OneHotEncoder


n_true = [320, 170, 510]

mu_true = [[5.0, 5.0],
           [6.5, 8.0],
           [9.5, 7.5]]

sigma_true = [[[1.0, 0.0],
               [0.0, 0.7]],
              [[2.0, -0.7],
               [-0.7, 1.0]],
              [[0.7, 0.9],
               [0.9, 5.0]]]


def generate_data(n, mu, sigma):
    X = []
    y = []
    for i, n_c in enumerate(n):
        X.append(mvn(mu[i], sigma[i]).rvs(n_c))
        y.append([i] * n_c)

    X = np.vstack(X)
    y = np.concatenate(y).reshape(-1, 1)

    return X, OneHotEncoder(categories='auto', sparse=False).fit_transform(y)


def plot_data(X, color='grey', alpha=0.7):
    plt.scatter(X[:,0], X[:,1], c=color, alpha=alpha)


def plot_densities(X, mu, sigma, alpha=0.5):
    grid_x, grid_y = np.mgrid[X[:,0].min():X[:,0].max():200j,
                     X[:,1].min():X[:,1].max():200j]
    grid = np.stack([grid_x, grid_y], axis=-1)

    for mu_c, sigma_c in zip(mu, sigma):
        plt.contour(grid_x, grid_y, mvn(mu_c, sigma_c).pdf(grid), colors='grey', alpha=alpha)
