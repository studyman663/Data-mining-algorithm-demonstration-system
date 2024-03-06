import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.spatial.distance import pdist, squareform
plt.rcParams['font.sans-serif'] = ['SimHei']
class myKmeans:
    def L2(self, vecXi, vecXj):
        return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))
    def Run(self, data, k):
        m = np.shape(data)[0]  # 样本总数
        sampleTag = np.zeros(m)
        n = np.shape(data)[1]  # 样本向量的特征数
        # 随机产生k个初始簇中心
        clusterCents = np.zeros((k, n))
        for j in range(n):
            minJ = min(data[:, j])
            rangeJ = float(max(data[:, j]) - minJ)
            clusterCents[:, j] = minJ + rangeJ * np.random.rand(k)
        sampleTagChanged = True
        num_center = 0.0
        while sampleTagChanged:  # 如果没有点发生分配结果改变，则结束
            sampleTagChanged = False
            num_center = 0.0
            # 计算每个样本点到各簇中心的距离
            for i in range(m):
                minD = np.inf
                minIndex = -1
                for j in range(k):
                    d = self.L2(clusterCents[j, :], data[i, :])
                    if d < minD:
                        minD = d
                        minIndex = j
                if sampleTag[i] != minIndex:
                    sampleTagChanged = True
                sampleTag[i] = minIndex
                num_center += minD ** 2
            # 重新计算簇中心
            for j in range(k):
                ClustI = data[np.nonzero(sampleTag == j)[0]]
                clusterCents[j, :] = np.mean(ClustI, axis=0)
            # 可视化聚类结果
            img_path = 'result/kmeans.png'
            plt.scatter(data[:, 0], data[:, 1], c=sampleTag)
            plt.scatter(clusterCents[:, 0], clusterCents[:, 1], c='r', marker='^', linewidths=7)
            plt.savefig(img_path)
            plt.close()
            return img_path

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
        plt.scatter(data[:, 0], data[:, 1], c=sampleTag)
        plt.scatter(clusterCents[:, 0], clusterCents[:, 1], c='r', marker='^', linewidths=7)
        plt.savefig(img_path)
        plt.close()
        return img_path
    
class myAgnes():
    def L2(self, vecXi, vecXj):
        return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))
    def linkage_matrix(sefl, X):
        return squareform(pdist(X, 'euclidean'))
    def Run(self, data, k):
        m = len(data)  # 样本总数
        clusters = [[i] for i in range(m)]  # 初始时每个样本都是一个簇
        linkage_mat = self.linkage_matrix(data)  # 计算距离矩阵
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
        img_path= 'result/agnes.png'
        cluster_centers = np.array([np.mean(data[cluster], axis=0) for cluster in clusters])
        colors = cm.rainbow(np.linspace(0, 1, len(clusters)))  # 使用彩虹色映射获取不同颜色值
        for i, cluster in enumerate(clusters):
            plt.scatter(data[cluster, 0], data[cluster, 1], color=colors[i])
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='r', marker='^', linewidths=7)
        plt.savefig(img_path)
        plt.close()
        # 返回最终的簇划分
        return img_path

class myDiana():
    def L2(self, vecXi, vecXj):
        return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))
    def linkage_matrix(sefl, X):
        return squareform(pdist(X, 'euclidean'))
    def Run(self, data, k):
        m = len(data)  # 样本总数
        clusters = [[i] for i in range(m)]  # 初始时每个样本都是一个簇
        linkage_mat = self.linkage_matrix(data)  # 计算距离矩阵
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
        img_path= 'result/diana.png'
        cluster_centers = np.array([np.mean(data[cluster], axis=0) for cluster in clusters])
        colors = cm.viridis(np.linspace(0, 1, len(clusters)))  # 使用彩虹色映射获取不同颜色值
        for i, cluster in enumerate(clusters):
            plt.scatter(data[cluster, 0], data[cluster, 1], color=colors[i])
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='r', marker='^', linewidths=7)
        plt.savefig(img_path)
        plt.close()
        # 返回最终的簇划分
        return img_path

class myDbscan():
    def __init__(self, eps, min_samples):
        self.eps = eps  # 邻域半径
        self.min_samples = min_samples  # 最小样本数

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum(np.power(point1 - point2, 2)))

    def region_query(self, data, point_index):
        neighbors = []
        for i, point in enumerate(data):
            if self.euclidean_distance(point, data[point_index]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, data, point_index, neighbors, cluster_id, visited, clusters):
        clusters[cluster_id].append(point_index)
        visited[point_index] = True

        i = 0
        while i < len(neighbors):
            neighbor_index = neighbors[i]
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                new_neighbors = self.region_query(data, neighbor_index)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            if neighbor_index not in clusters[cluster_id]:
                clusters[cluster_id].append(neighbor_index)
            i += 1

    def Run(self, data):
        m = np.shape(data)[0]  # 样本总数
        visited = np.zeros(m, dtype=bool)  # 样本是否已访问
        clusters = []  # 聚类结果
        cluster_id = 0  # 聚类标识

        for i in range(m):
            if visited[i]:
                continue
            neighbors = self.region_query(data, i)
            if len(neighbors) < self.min_samples:
                visited[i] = True
            else:
                clusters.append([])
                self.expand_cluster(data, i, neighbors, cluster_id, visited, clusters)
                cluster_id += 1

        return clusters

    def visualize_clusters(self, data, clusters):
        cmap = cm.get_cmap('tab10')
        num_colors = len(clusters)
        for i, cluster in enumerate(clusters):
            cluster_points = data[cluster]
            color = cmap(i / num_colors)  # 根据聚类编号选择颜色
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color])

            cluster_center = np.mean(cluster_points, axis=0)
            plt.scatter(cluster_center[0], cluster_center[1], marker='^', color='r', linewidths=7)
        img_path= 'result/dbscan.png'
        plt.savefig(img_path)
        plt.close()
        return img_path