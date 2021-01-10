import numpy as np

from sklearn.cluster import KMeans, DBSCAN

# from helpers import plot_PCA

# TODO add PCA to extract most important features before clustering
# TODO add implementation of kmeans based on cosine similarity


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


def dbscan_process(x, y, num_of_clusters):
    np.random.seed(5)

    metrics = ['cosine']
    min_group = len(x) // num_of_clusters
    average_group_size = len(x) / num_of_clusters
    epsilon = [500, 100, 50, 10, 5] + [(i + 1)/1000 for i in range(1000)]
    was_resolution_found = False

    results = []
    for m in metrics:
        for i in range(min_group, 0, -1):
            for j in epsilon:
                model = DBSCAN(eps=j, min_samples=i, metric=m)
                result = model.fit(x)
                labels = result.labels_  #map(lambda z: z + 1, result.labels_)
                count_of_generated_clusters = len(set(labels))

                if count_of_generated_clusters == num_of_clusters:
                    results.append(labels)
                    was_resolution_found = True

    if not was_resolution_found:
        print("dbscan was not able to find parameters to create expected number of clusters")
        return

    se = []
    for result in results:
        se.append(calculate_squared_error(result, average_group_size))

    min_se_index = se.index(min(se))

    labels = results[min_se_index]
    labels = map(lambda z: z+1, labels)
    ret = list(zip(labels, y))
    ret.sort()
    return ret


def calculate_squared_error(labels, mean):
    tmp_dict = {}

    for i in labels:
        tmp_dict[i] = 0

    for i in labels:
        tmp_dict[i] += 1

    se = 0
    for i in tmp_dict.values():
        se += (i - mean) ** 2

    return se


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
