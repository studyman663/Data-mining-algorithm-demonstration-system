import numpy as np
from matplotlib import pyplot as plt


class myKmedoid():
    def L2(self,vecXi, vecXj):
        return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))
    def Run(self,data,k):
        m = np.shape(data)[0]  # 样本总数
        sampleTag = np.zeros(m)  # 样本对应的簇标记
        clusterCents = np.zeros((k, np.shape(data)[1]))  # 各簇的medoid
        num_center = 0.0  # 误差平方和
        # 随机选择k个medoid
        medoids = np.random.choice(m, k, replace=False)
        clusterCents = data[medoids]
        sampleTagChanged = True
        while sampleTagChanged:
            sampleTagChanged = False
            # 计算每个样本点到各medoid的距离
            distances = np.zeros((m, k))
            for i in range(m):
                for j in range(k):
                    distances[i, j] = self.L2(data[i], clusterCents[j])
            # 分配样本点到最近的medoid
            for i in range(m):
                minDist = np.min(distances[i])
                minIndex = np.where(distances[i] == minDist)[0][0]
                if sampleTag[i] != minIndex:
                    sampleTagChanged = True
                sampleTag[i] = minIndex
            # 更新簇的medoid
            for j in range(k):
                ClustI = data[np.nonzero(sampleTag == j)[0]]
                if len(ClustI) == 0:
                    continue
                # 计算成为medoid的代价
                medoidCosts = np.zeros(len(ClustI))
                for i, point in enumerate(ClustI):
                    totalDist = 0
                    for other in ClustI:
                        totalDist += self.L2(point, other)
                    medoidCosts[i] = totalDist
                # 选择代价最小的点作为新的medoid
                newMedoidIndex = np.argmin(medoidCosts)
                clusterCents[j] = ClustI[newMedoidIndex]

            # 计算误差平方和
            for i in range(m):
                num_center += np.min(distances[i]) ** 2

        # 可视化聚类结果
        img_path = 'result/kmedoid.png'
        plt.scatter(clusterCents[:, 0], clusterCents[:, 1], c='r', marker='^', linewidths=7)
        plt.scatter(data[:, 0], data[:, 1], c=sampleTag, linewidths=np.power(sampleTag + 0.5, 2))
        plt.savefig(img_path)
        return img_path