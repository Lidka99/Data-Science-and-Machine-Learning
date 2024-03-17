import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# from yellowbrick.cluster import KElbowVisualizer, silhouette_visualizer


variants = {
    'single': ['euclidean', 'manhattan', 'cosine'],  # 'l1', 'l2'
    'complete': ['euclidean', 'manhattan', 'cosine'],
    'average': ['euclidean', 'manhattan', 'cosine'],
    'ward': ['euclidean']
}
n_clusters_range = np.arange(5, 1, -1)


def agglomerative_grouping(dataset):
    results = [evaluate(dataset, metrics.silhouette_score, 'Silhouette score'),
               evaluate(dataset, metrics.calinski_harabasz_score, 'Calinski Harabasz score'),
               evaluate(dataset, metrics.davies_bouldin_score, 'Davies bouldin score')]

    j = 0
    for k in n_clusters_range:
        print()
        print('number of clusters: ', k)
        print('variant; Silhouette score; Calinski Harabasz score; Davies bouldin score')

        for i in range(len(results[0])):
            print(results[0][i][0], ' ', results[0][i][1], '; ', results[0][i][2][j], '; ',
                  results[1][i][2][j], '; ', results[2][i][2][j])
        j += 1

    # #Best number of clusters
    # max_result = 0   # znajdowanie największej wartości ze wszystkich
    #
    # for result in results:
    #     if max_result < result[2]:
    #         max_result = result[2]
    #
    # best_results = []
    # for result in results:
    #     if result[2] == max_result:
    #         best_results.append(result)
    #
    # print()
    # print('best: ', best_results)
    #
    # for result in best_results:
    #     y = AgglomerativeClustering(
    #         n_clusters=result[3], affinity=result[1],
    #         linkage=result[0]).fit_predict(dataset)
    #     print()
    #     print('linkage: ', result[0], '    affinity: ', result[1], '   number of clusters: ', result[3])
    #     print('Calinski Harabasz index ', metrics.calinski_harabasz_score(dataset, y))
    #
    #     print('Silhouette score ', metrics.silhouette_score(dataset, y))
    #
    #     print('davies bouldin score ', metrics.davies_bouldin_score(dataset, y))

    # for variant_id, linkage in zip(range(len(variants)), variants.keys()):
    #
    #     for affinity in variants[linkage]:
    #
    #         model = AgglomerativeClustering(affinity=affinity,
    #                 linkage=linkage)
    #
    #         visualizer = KElbowVisualizer(model, k=(2, 15), metric='calinski_harabasz', timings=True)
    #         visualizer.fit(dataset)  # Fit the data to the visualizer
    #         visualizer.show()  # Finalize and render the figure

    # davies-bouldin


def evaluate(dataset, function, ylabel):
    results = []

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for variant_id, linkage in zip(range(len(variants)), variants.keys()):
        plt.subplot(2, 2, variant_id + 1, title="linkage: " + linkage)
        plt.grid()
        plt.xlabel('N clusters')
        plt.ylabel(ylabel)
        for affinity in variants[linkage]:
            score = []
            max_score = 0
            max_score_clusters = 0
            for n_clusters in n_clusters_range:
                y = AgglomerativeClustering(
                    n_clusters=n_clusters, affinity=affinity,
                    linkage=linkage).fit_predict(dataset)
                # silhouette score
                scr = function(dataset, y)
                score.append(scr)
                if scr > max_score:
                    max_score = scr
                    max_score_clusters = n_clusters

            results.append([linkage, affinity, score])

            plt.plot(n_clusters_range,
                     score,
                     label=affinity)
            plt.legend()
    plt.show()

    return results


def visualizer(dataset, x_index, y_index, z_index, x_label, y_label, z_label, dataset_name):
    for variant_id, linkage in zip(range(len(variants)), variants.keys()):
        for affinity in variants[linkage]:
            for n_clusters in n_clusters_range:
                plot_visualisation(dataset, n_clusters, linkage, affinity,
                                   linkage + ' ' + affinity + ' ' + str(n_clusters) + ' clusters', x_index, y_index,
                                   z_index,
                                   x_label, y_label, z_label)


def plot_visualisation(data_set, n_clusters, linkage, affinity,
                       title, x_index, y_index, z_index, x_label, y_label, z_label):
    #pca_ = PCA(n_components=3)
    #X_Demo_fit_pca = pca_.fit_transform(data_set)
    X_Demo_fit_pca = data_set

    model = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    labels = model.fit_predict(X_Demo_fit_pca)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_Demo_fit_pca[:, x_index], X_Demo_fit_pca[:, y_index], X_Demo_fit_pca[:, z_index],
               c=labels, cmap='viridis',
               edgecolor='k', s=40, alpha=0.5)

    ax.set_title("Trzy wybrane kolumny - " + title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.dist = 10
    plt.show()
