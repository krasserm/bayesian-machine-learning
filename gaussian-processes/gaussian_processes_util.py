import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D


# ------------------------------------------
#  GPs for regression utils
# ------------------------------------------


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()


def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)


# ------------------------------------------
#  GPs for classification utils
# ------------------------------------------


def plot_data_1D(X, t):
    class_0 = t == 0
    class_1 = t == 1

    plt.scatter(X[class_1], t[class_1], label='Class 1', marker='x', color='red')
    plt.scatter(X[class_0], t[class_0], label='Class 0', marker='o', edgecolors='blue', facecolors='none')


def plot_data_2D(X, t):
    class_1 = np.ravel(t == 1)
    class_0 = np.ravel(t == 0)

    plt.scatter(X[class_1, 0], X[class_1, 1], label='Class 1', marker='x', c='red')
    plt.scatter(X[class_0, 0], X[class_0, 1], label='Class 0', marker='o', edgecolors='blue', facecolors='none')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')


def plot_pt_2D(grid_x, grid_y, grid_z):
    plt.contourf(grid_x, grid_y, grid_z, cmap='plasma', alpha=0.3, levels=np.linspace(0, 1, 11))
    plt.colorbar(format='%.2f')


def plot_db_2D(grid_x, grid_y, grid_z, decision_boundary=0.5):
    levels = [decision_boundary]
    cs = plt.contour(grid_x, grid_y, grid_z, levels=levels, colors='black', linestyles='dashed', linewidths=2)
    plt.clabel(cs, fontsize=20)


# ------------------------------------------
#  Sparse GP utils
# ------------------------------------------


def generate_animation(theta_steps, X_m_steps, X_test, f_true, X, y, sigma_y, phi_opt, q, interval=100):
    fig, ax = plt.subplots()

    line_func, = ax.plot(X_test, f_true, label='Latent function', c='k', lw=0.5)
    pnts_ind = ax.scatter([], [], label='Inducing variables', c='m')

    line_pred, = ax.plot([], [], label='Prediction', c='b')
    area_pred = ax.fill_between([], [], [], label='Epistemic uncertainty', color='r', alpha=0.1)

    ax.set_title('Optimization of a sparse Gaussian process')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-3, 3.5)
    ax.legend(loc='upper right')

    def plot_step(i):
        theta = theta_steps[i]
        X_m = X_m_steps[i]

        mu_m, A_m, K_mm_inv = phi_opt(theta, X_m, X, y, sigma_y)
        f_test, f_test_cov = q(X_test, theta, X_m, mu_m, A_m, K_mm_inv)
        f_test_var = np.diag(f_test_cov)
        f_test_std = np.sqrt(f_test_var)

        ax.collections.clear()
        pnts_ind = ax.scatter(X_m, mu_m, c='m')

        line_pred.set_data(X_test, f_test.ravel())
        area_pred = ax.fill_between(X_test.ravel(),
                                    f_test.ravel() + 2 * f_test_std,
                                    f_test.ravel() - 2 * f_test_std,
                                    color='r', alpha=0.1)

        return line_func, pnts_ind, line_pred, area_pred

    result = animation.FuncAnimation(fig, plot_step, frames=len(theta_steps), interval=interval)

    # Prevent output of last frame as additional plot
    plt.close()

    return result

