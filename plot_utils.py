import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import scipy.stats as stats
from sklearn import mixture

def plot_gmm(ax, data, bounds, n_samp=100j, n_comp=5):
    clf = mixture.GaussianMixture(n_components=n_comp, covariance_type="full", n_init=1, random_state=2541)
    clf.fit(data)

    X, Y = np.mgrid[bounds[0]:bounds[1]:n_samp, bounds[2]:bounds[3]:n_samp]
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX).reshape(X.shape)

    cmap = cm.Oranges_r(np.linspace(0,1,50))
    cmap = colors.ListedColormap(cmap[30:,:-1][::1])

    ax.contourf(X, Y, Z, levels=np.logspace(0, 1.1, 15), cmap=cmap)

def plot_kde(ax, data, bounds, n_samp=100j):
    xx, yy = np.mgrid[bounds[0]:bounds[1]:n_samp, bounds[2]:bounds[3]:n_samp]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([data[:, 0], data[:, 1]])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    cfset = ax.contourf(xx, yy, f, cmap='Blues')

def get_emp_covar_3d(data):
    cv = np.cov(data.T)
    cov12 = [[cv[0, 0], cv[0, 1]], [cv[1, 0], cv[1, 1]]]
    cov13 = [[cv[0, 0], cv[0, 2]], [cv[2, 0], cv[2, 2]]]
    cov23 = [[cv[1, 1], cv[1, 2]], [cv[2, 1], cv[2, 2]]]

    return cov12, cov13, cov23

def plot_mv_gaussian(ax, mean, cov, cmap, bounds):
    x_range = np.linspace(bounds[0], bounds[1], 200)
    y_range = np.linspace(bounds[2], bounds[3], 200)

    x, y = np.meshgrid(x_range, y_range)
    pos = np.dstack((x, y))

    dist = stats.multivariate_normal(mean, cov)
    z = dist.pdf(pos)

    ax.contour(x, y, z, 5, cmap=cmap, alpha=1, zorder=-100)


def plot_data_contour(ax, data, colours=["Blues", "Oranges", "Greens"]):
    x_max = np.max(data[:, 0:2]) - 1
    x_min = np.min(data[:, 0:2]) + 1
    y_max = np.max(data[:, 1:3]) - 1
    y_min = np.min(data[:, 1:3]) + 1

    bounds = [x_min, x_max, y_min, y_max]

    means = np.mean(data, axis=0)
    covs = get_emp_covar_3d(data)
    plot_mv_gaussian(ax, means[[0, 1]], covs[0], colours[0], bounds)
    plot_mv_gaussian(ax, means[[0, 2]], covs[1], colours[1], bounds)
    plot_mv_gaussian(ax, means[[1, 2]], covs[2], colours[2], bounds)

    lines = [Line2D([0], [0], color=plt.get_cmap(colours[0])(0.5), lw=3),
             Line2D([0], [0], color=plt.get_cmap(colours[1])(0.5), lw=3),
             Line2D([0], [0], color=plt.get_cmap(colours[2])(0.5), lw=3)]

    ax.legend(lines, ['1v2', '1v3', '2v3'])


def viz_data(data, bounds=4):
    f, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=2)
    ax[0].set_xlabel("X0")
    ax[0].set_ylabel("X1")
    ax[0].set_ylim(-bounds, bounds)
    ax[0].set_xlim(-bounds, bounds)

    ax[1].scatter(data[:, 0], data[:, 2], alpha=0.5, s=2)
    ax[1].set_xlabel("X0")
    ax[1].set_ylabel("X2")
    ax[1].set_ylim(-bounds, bounds)
    ax[1].set_xlim(-bounds, bounds)

    ax[2].scatter(data[:, 1], data[:, 2], alpha=0.5, s=2)
    ax[2].set_xlabel("X1")
    ax[2].set_ylabel("X2")
    ax[2].set_ylim(-bounds, bounds)
    ax[2].set_xlim(-bounds, bounds)
    f.tight_layout()
    plt.show()


def viz_err(pred, true, ax):
    bounds = 4
    f, ax = plt.subplots(1, 3, figsize=(15, 4))

    d1 = pred[:, 0]
    d2 = pred[:, 1]
    d3 = pred[:, 2]

    plot_gmm(ax[0], true[:, 0:2], bounds=[-bounds, bounds, -bounds, bounds], n_comp=6)
    ax[0].scatter(d1, d2, s=4, alpha=0.5, label="Samples", c='k')
    ax[0].legend()
    ax[0].set_xlabel("X0")
    ax[0].set_ylabel("X1")
    ax[0].set_ylim(-bounds, bounds)
    ax[0].set_xlim(-bounds, bounds)

    plot_gmm(ax[1], true[:, [0, 2]], bounds=[-bounds, bounds, -bounds, bounds], n_comp=8)
    ax[1].scatter(d1, d3, s=4, alpha=0.5, label="Samples", c='k')
    ax[1].legend()
    ax[1].set_xlabel("X0")
    ax[1].set_ylabel("X2")
    ax[1].set_ylim(-bounds, bounds)
    ax[1].set_xlim(-bounds, bounds)

    plot_gmm(ax[2], true[:, 1:3], bounds=[-bounds, bounds, -bounds, bounds], n_comp=10)
    ax[2].scatter(d2, d3, s=4, alpha=0.5, label="Samples", c='k')
    ax[2].legend()
    ax[2].set_xlabel("X1")
    ax[2].set_ylabel("X2")
    ax[2].set_ylim(-bounds, bounds)
    ax[2].set_xlim(-bounds, bounds)
