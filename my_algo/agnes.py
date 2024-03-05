import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_blobs


def L2(vecXi, vecXj):
    '''
    计算欧氏距离
    '''
    return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))


def linkage_matrix(X):
    '''
    计算所有数据点之间的距离矩阵
    '''
    return squareform(pdist(X, 'euclidean'))


def agnes(S, k):
    '''
    AGNES层次聚类算法
    '''
    m = len(S)  # 样本总数
    clusters = [[i] for i in range(m)]  # 初始时每个样本都是一个簇
    linkage_mat = linkage_matrix(S)  # 计算距离矩阵

    while len(clusters) > k:
        # 计算簇之间的距离
        distances = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # 使用平均距离作为簇间距离
                dist = np.mean([linkage_mat[x][y] for x in clusters[i] for y in clusters[j]])
                distances.append((dist, i, j))
        distances.sort()  # 按距离排序

        # 合并最近的簇对
        dist, i, j = distances[0]
        clusters[i].extend(clusters[j])
        clusters.pop(j)

        # 更新距离矩阵
        for ci in clusters:
            for cj in clusters:
                if ci != cj:
                    new_dist = np.mean([linkage_mat[x][y] for x in ci for y in cj])
                    linkage_mat[ci[0]][cj[0]] = new_dist
                    linkage_mat[cj[0]][ci[0]] = new_dist

        # 可视化聚类结果
    cluster_centers = np.array([np.mean(S[cluster], axis=0) for cluster in clusters])
    plt.scatter(S[:, 0], S[:, 1], c='gray')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='r', marker='^', linewidths=7)
    plt.show()

    # 返回最终的簇划分
    return clusters



# 示例数据
X,y=make_blobs(n_samples=100,centers=4,random_state=42)
clusters = agnes(X, 3)