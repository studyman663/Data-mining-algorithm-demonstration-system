from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成聚类数据
X, y = make_blobs(n_samples=300, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.5])

# 创建KMedoids实例
kmedoids = KMedoids(n_clusters=4, random_state=0)

# 训练模型
kmedoids.fit(X)

# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=kmedoids.labels_, s=50, cmap='viridis')
# plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], marker='^', s=200, c='red')
plt.show()