import numpy as np
import matplotlib.pyplot as plt


def L2(vecXi, vecXj):
    return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))


def kMedoids(S, k):
    '''
    K-Medoids聚类
    '''
    m = np.shape(S)[0]  # 样本总数
    sampleTag = np.zeros(m)  # 样本对应的簇标记
    clusterCents = np.zeros((k, np.shape(S)[1]))  # 各簇的medoid
    SSE = 0.0  # 误差平方和

    # 随机选择k个medoid
    medoids = np.random.choice(m, k, replace=False)
    clusterCents = S[medoids]

    sampleTagChanged = True
    while sampleTagChanged:
        sampleTagChanged = False

        # 计算每个样本点到各medoid的距离
        distances = np.zeros((m, k))
        for i in range(m):
            for j in range(k):
                distances[i, j] = L2(S[i], clusterCents[j])

        # 分配样本点到最近的medoid
        for i in range(m):
            minDist = np.min(distances[i])
            minIndex = np.where(distances[i] == minDist)[0][0]
            if sampleTag[i] != minIndex:
                sampleTagChanged = True
            sampleTag[i] = minIndex

        # 更新簇的medoid
        for j in range(k):
            ClustI = S[np.nonzero(sampleTag == j)[0]]
            if len(ClustI) == 0:
                continue

            # 计算成为medoid的代价
            medoidCosts = np.zeros(len(ClustI))
            for i, point in enumerate(ClustI):
                totalDist = 0
                for other in ClustI:
                    totalDist += L2(point, other)
                medoidCosts[i] = totalDist

            # 选择代价最小的点作为新的medoid
            newMedoidIndex = np.argmin(medoidCosts)
            clusterCents[j] = ClustI[newMedoidIndex]

        # 计算误差平方和
        for i in range(m):
            SSE += np.min(distances[i]) ** 2

        # 可视化聚类结果
        plt.scatter(clusterCents[:, 0], clusterCents[:, 1], c='r', marker='^', linewidths=7)
        plt.scatter(S[:, 0], S[:, 1], c=sampleTag, linewidths=np.power(sampleTag + 0.5, 2))
        plt.show()
        print(SSE)

    return sampleTag, clusterCents, SSE


# 示例数据
samples = np.loadtxt("kmeansSamples.txt")
k = 3

# 运行K-Medoids算法
sampleTag, clusterCents, SSE = kMedoids(samples, k)