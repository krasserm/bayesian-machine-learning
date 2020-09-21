import daft
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


def plot_densities(X, mu, sigma, alpha=0.5, colors='grey'):
    grid_x, grid_y = np.mgrid[X[:,0].min():X[:,0].max():200j,
                     X[:,1].min():X[:,1].max():200j]
    grid = np.stack([grid_x, grid_y], axis=-1)

    for mu_c, sigma_c in zip(mu, sigma):
        plt.contour(grid_x, grid_y, mvn(mu_c, sigma_c).pdf(grid), colors=colors, alpha=alpha)


def plot_gmm_plate(filename="gmm.png", dpi=100):
    pgm = daft.PGM([3.0, 2.5], origin=(0, 0))
    pgm.add_node(daft.Node("theta", r"$\mathbf{\theta}$", 1, 2, fixed=True))
    pgm.add_node(daft.Node("ti", r"$\mathbf{t}_i$", 1, 1))
    pgm.add_node(daft.Node("xi", r"$\mathbf{x}_i$", 2, 1, observed=True))
    pgm.add_edge("theta", "ti")
    pgm.add_edge("theta", "xi")
    pgm.add_edge("ti", "xi")
    pgm.add_plate(daft.Plate([0.4, 0.5, 2.2, 1.0], label=r"$N$"))
    ax = pgm.render()
    ax.text(0.8, 0.5, 'Gaussian mixture model')
    pgm.savefig(filename, dpi=dpi)
