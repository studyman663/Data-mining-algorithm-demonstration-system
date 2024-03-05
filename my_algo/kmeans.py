import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def L2(vecXi, vecXj):
    '''
    计算欧氏距离
    para vecXi：点坐标，向量
    para vecXj：点坐标，向量
    retrurn: 两点之间的欧氏距离
    '''
    return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))


def kMeans(S, k, distMeas=L2):
    '''
    K均值聚类
    para S：样本集，多维数组
    para k：簇个数
    para distMeas：距离度量函数，默认为欧氏距离计算函数
    return sampleTag：一维数组，存储样本对应的簇标记
    return clusterCents：二维数组，各簇中心
    return SSE:误差平方和
    '''
    m = np.shape(S)[0]  # 样本总数
    sampleTag = np.zeros(m)
    n = np.shape(S)[1]  # 样本向量的特征数

    # 随机产生k个初始簇中心
    clusterCents = np.zeros((k, n))
    for j in range(n):
        minJ = min(S[:, j])
        rangeJ = float(max(S[:, j]) - minJ)
        clusterCents[:, j] = minJ + rangeJ * np.random.rand(k)

    sampleTagChanged = True
    SSE = 0.0
    while sampleTagChanged:  # 如果没有点发生分配结果改变，则结束
        sampleTagChanged = False
        SSE = 0.0

        # 计算每个样本点到各簇中心的距离
        for i in range(m):
            minD = np.inf
            minIndex = -1
            for j in range(k):
                d = distMeas(clusterCents[j, :], S[i, :])
                if d < minD:
                    minD = d
                    minIndex = j
            if sampleTag[i] != minIndex:
                sampleTagChanged = True
            sampleTag[i] = minIndex
            SSE += minD ** 2

        # 重新计算簇中心
        for j in range(k):
            ClustI = S[np.nonzero(sampleTag == j)[0]]
            clusterCents[j, :] = np.mean(ClustI, axis=0)

        # 可视化聚类结果


    return sampleTag, clusterCents, SSE


# 示例数据
X,y=make_blobs(n_samples=50,centers=4,random_state=42)

# 运行K-Means算法
sampleTag, clusterCents, SSE = kMeans(X, 3)

plt.scatter(X[:, 0], X[:, 1], c=sampleTag, linewidths=np.power(sampleTag + 0.5, 2))
plt.scatter(clusterCents[:, 0], clusterCents[:, 1], c='r', marker='^', linewidths=7)
plt.show()