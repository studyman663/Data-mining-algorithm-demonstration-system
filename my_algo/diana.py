import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

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

def diana(S, k):
    '''
    DIANA层次聚类算法
    '''
    m = len(S)  # 样本总数
    clusters = [[i for i in range(m)]] # 初始时所有样本属于一个簇
    linkage_mat = linkage_matrix(S)  # 计算距离矩阵

    while len(clusters) < k:  # 当簇的数量大于k时继续分裂
        print('y')
        # 寻找距离最远的样本对
        max_dist = -1  # 初始化为一个不可能的值
        split_indices = None
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                if len(clusters[i]) > 0 and len(clusters[j]) > 0:  # 确保簇不为空
                    dist = linkage_mat[clusters[i][0], clusters[j][0]]  # 注意这里是矩阵的索引方式
                    print(dist)
                    if dist > max_dist:
                        max_dist = dist
                        split_indices = (i, j)
        print(split_indices)
        # 分裂簇
        i, j = split_indices
        clusters[j] = clusters[j] + clusters[i]  # 正确合并簇j和簇i
        clusters.pop(i)  # 删除空簇i

        # 更新距离矩阵是不必要的，因为我们已经知道哪些点属于哪个簇
        # 如果需要可视化，可以在这里进行

    # 可视化最终的簇划分（这里只在最后进行一次可视化）
    if len(clusters) > 0:  # 确保clusters不为空
        cluster_centers = np.array([np.mean(S[np.array(cluster)], axis=0) for cluster in clusters])
        plt.scatter(S[:, 0], S[:, 1], c='gray')  # 绘制所有样本点
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='r', marker='^', linewidths=7)  # 绘制簇中心点
        plt.show()

    # 返回最终的簇划分
    return clusters

# 示例数据
samples = np.loadtxt("kmeansSamples.txt")
k = 3  # 预期的簇数量

# 运行DIANA算法
clusters = diana(samples, k)
print("最终的簇划分:", clusters)