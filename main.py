import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.40)

clf = SVC(kernel='linear')
clf.fit(X, Y)

def plot_svc_decision_boundary(clf, X, ax=None, plot_support=True):
    """Plot the decision boundary of an SVC."""
    if ax is None:
        ax = plt.gca()
    xlim = (-1, 3.5)
    ylim = (-1, 5)
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='k')
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
plot_svc_decision_boundary(clf, X)
plt.show()
