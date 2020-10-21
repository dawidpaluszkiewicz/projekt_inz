import numpy as np

from sklearn.cluster import KMeans, DBSCAN


def kmean_process(x, y, num_of_clusters):
    np.random.seed(5)

    est = KMeans(n_clusters=num_of_clusters, random_state=0, n_init=100, max_iter=1000).fit(x)

    labels = est.labels_

    ret = list(zip(labels, y))
    ret.sort()
    return ret


def dbscan_process(x, y):  # TODO uzgodnic z prowadzacym
    X = np.array(x)
    np.random.seed(5)

    clustering = DBSCAN(eps=0, min_samples=0).fit(X)
    labels = clustering.labels_

    ret = list(zip(labels, y)).sort()
    return ret
