from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# 生成样本数据
X, y_true = make_blobs(random_state=1)

# 创建并配置AGNES聚类器
agg = AgglomerativeClustering(n_clusters=3)

# 拟合聚类模型
agg.fit(X)

# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=agg.labels_, s=50, cmap='viridis')
# plt.scatter(agg.cluster_centers_[:, 0], agg.cluster_centers_[:, 1], marker='^', s=200, c='red')
plt.show()