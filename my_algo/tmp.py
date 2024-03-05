import random
import matplotlib.pyplot as plt


# 产生随机数据点
def generate_data(n):
    points = []
    for i in range(n):
        x = random.random()
        y = random.random()
        points.append([x, y])
    return points


# 计算两个点之间的距离
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# 计算簇内平均距离
def avg_dist(cluster):
    sum = 0
    num = len(cluster)
    for i in range(num - 1):
        for j in range(i + 1, num):
            sum += distance(cluster[i], cluster[j])
    return sum / (num * (num - 1) / 2)


# Diana算法
def diana(points, k):
    # 初始化所有点为一个簇
    clusters = [points]

    while len(clusters) > k:

        # 找出直径最大的簇
        max_cluster = clusters[0]
        max_diameter = avg_dist(clusters[0])
        for c in clusters:
            diameter = avg_dist(c)
            if diameter > max_diameter:
                max_cluster = c
                max_diameter = diameter

        # 从最大簇中分离出直径最大的点作为新簇
        splinter = max_cluster[0]
        max_dist = 0
        for p in max_cluster:
            dist = distance(p, splinter)
            if dist > max_dist:
                splinter = p
                max_dist = dist

        # 将点放入新簇
        splinter_cluster = [splinter]

        # 将其他点分到新簇或原簇
        for p in max_cluster:
            if p != splinter:
                if distance(p, splinter) < distance(p, clusters):
                    splinter_cluster.append(p)
                else:
                    clusters.remove(max_cluster)
                    clusters.append(p)

        # 将新簇加入原集合
        clusters.append(splinter_cluster)

    return clusters


# 绘图显示结果
def plot_clusters(clusters):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]
        for point in cluster:
            plt.plot(point[0], point[1], color + 'o')
    plt.show()


if __name__ == '__main__':
    # 生成100个数据点
    points = generate_data(100)
    print(type(points))

    # Diana算法分成6个簇
    clusters = diana(points, 6)

    # 绘制结果
    plot_clusters(clusters)