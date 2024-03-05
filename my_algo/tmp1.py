import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs


def diana_clustering(X, num_clusters, max_iter=100):
    # 初始化簇
    clusters = [X]
    splinter_group = []
    old_party = []
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    for iteration in range(max_iter):
        # 寻找最大直径的簇
        max_diam_cluster = max(clusters, key=lambda c: np.max(pdist(c)))

        # 计算簇内所有点的平均相异度
        avg_dissimilarities = np.mean(pdist(max_diam_cluster), axis=0)

        # 找到平均相异度最大的点
        p = max_diam_cluster[np.argmax(avg_dissimilarities)]

        # 分割簇
        splinter_group.append(p)
        old_party.extend(max_diam_cluster[max_diam_cluster != p])
        clusters.remove(max_diam_cluster)

        # 寻找要加入splinter_group的点
        while True:
            # 计算old_party中每个点到splinter_group的最近距离
            distances = squareform(pdist(np.vstack([splinter_group, old_party]), 'euclidean'))[:, :len(splinter_group)]
            min_distances_to_splinter = np.min(distances, axis=1)

            # 计算old_party中每个点到其簇的最近距离
            min_distances_to_old = np.min(pdist(old_party), axis=1)

            # 找到要加入splinter_group的点
            point_to_move = old_party[min_distances_to_splinter <= min_distances_to_old]
            if len(point_to_move) == 0:
                break
            splinter_group.extend(point_to_move)
            old_party = np.setdiff1d(old_party, point_to_move)

        # 形成新的簇
        new_cluster = np.vstack([splinter_group])
        clusters.append(new_cluster)
        splinter_group = []

        # 检查簇的数量是否达到要求
        if len(clusters) == num_clusters:
            break

    # 可视化结果
    plt.figure(figsize=(12, 8))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i + 1}', alpha=0.5)
    plt.title('DIANA Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


# 生成示例数据
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)

# 运行DIANA算法
diana_clustering(X, num_clusters=5)