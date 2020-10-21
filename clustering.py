import numpy as np

from sklearn.cluster import KMeans, DBSCAN

# from helpers import plot_PCA


def kmean_process(x, y, num_of_clusters):
    np.random.seed(5)

    est = KMeans(n_clusters=num_of_clusters, random_state=0, n_init=100, max_iter=1000).fit(x)

    labels = est.labels_

    ret = list(zip(labels, y))
    ret.sort()
    return ret


def kmean_process_equal_clusters(x, y, num_of_clusters):
    np.random.seed(5)

    est = KMeans(n_clusters=num_of_clusters, random_state=0, n_init=100, max_iter=1000).fit(x)

    labels = est.labels_
    max_num_of_elements_in_cluster = labels.size / num_of_clusters
    cluster_centres = est.cluster_centers_

    distances_labels = compute_distances_from_centers(x, y, cluster_centres)

    sorted_data_by_cluster_center = []
    for i in range(cluster_centres.shape[0]):
        sorted_data_by_cluster_center.append(sorted(distances_labels, key=lambda p: p[1][i]))

    ret = []
    to_plot = []
    for i in range(len(sorted_data_by_cluster_center[0])):
        exceeding = check_if_cluster_is_full(ret, max_num_of_elements_in_cluster)
        index = get_nearest_index(sorted_data_by_cluster_center, exceeding)

        value = sorted_data_by_cluster_center[index][0]
        ret.append((index, value[0]))
        to_plot.append((index, value[2]))
        for j in range(len(sorted_data_by_cluster_center)):
            sorted_data_by_cluster_center[j].remove(value)

    # to_plot.sort(key=(lambda k: k[0]))  TODO think about better way to visualize
    # plot_PCA(to_plot, cluster_centres)

    ret.sort()
    return ret


def dbscan_process(x, y, num_of_clusters):  # TODO works very poorly, to adjust later
    np.random.seed(5)

    admissible_size_of_cluster = 2  # no reason to choose 3, to change in future
    labels = None
    for i in range(1, 2):
        clustering = DBSCAN(eps=i, min_samples=admissible_size_of_cluster).fit(x)
        labels = clustering.labels_
        count_of_clusters = len(set(labels))
        tmp = {}
        for j in set(labels):
            tmp[j] = 0

        for j in labels:
            tmp[j] += 1
        print(count_of_clusters, i, tmp)
        # if count_of_clusters == num_of_clusters:
        #     break
    else:
        print("Could not find a proper clustering for dbscan")

    ret = list(zip(labels, y))
    ret.sort()
    return ret


def compute_distances_from_centers(x, y, cluster_centres):
    ret = []

    # element, label
    for e, l in zip(x, y):
        centers = []
        for c in cluster_centres:
            centers.append(np.sqrt(np.sum(np.square(e - c))))
        ret.append((l, centers, e))
    return ret


def get_nearest_index(arr, to_skip):
    index = -1
    min_value = 9999
    for i in range(len(arr)):
        if i not in to_skip:
            if arr[i][0][1][i] < min_value:
                index = i
                min_value = arr[i][0][1][i]

    return index


def check_if_cluster_is_full(curr_result, max_num):
    to_check = {}
    for i in curr_result:
        to_check[i[0]] = 0

    for i in curr_result:
        to_check[i[0]] += 1

    exceeding = []
    for key in to_check.keys():
        if to_check[key] >= max_num:
            exceeding.append(key)

    return exceeding
