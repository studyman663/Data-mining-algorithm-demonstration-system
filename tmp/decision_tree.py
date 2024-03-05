import numpy as np
import matplotlib.pyplot as plt

# 定义决策树节点类
import pydotplus
from sklearn import tree
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz
from IPython.display import Image
import dtreeviz

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_index = feature_index  # 特征索引
        self.threshold = threshold  # 分割阈值
        self.value = value  # 叶节点值
        self.true_branch = true_branch  # 左子树
        self.false_branch = false_branch  # 右子树

# 定义决策树分类器类
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # 树的最大深度

    # 计算基尼指数
    def gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini_index = 1 - np.sum(probabilities**2)
        return gini_index

    # 根据特征和分割阈值将数据集划分为两个子集
    def split_data(self, X, y, feature_index, threshold):
        mask = X[:, feature_index] <= threshold
        X_true = X[mask]
        y_true = y[mask]
        X_false = X[~mask]
        y_false = y[~mask]
        return X_true, y_true, X_false, y_false

    # 寻找最佳分割特征和阈值
    def find_best_split(self, X, y):
        best_gini = 1
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_true, y_true, X_false, y_false = self.split_data(X, y, feature_index, threshold)
                gini = (len(y_true) * self.gini(y_true) + len(y_false) * self.gini(y_false)) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    # 递归构建决策树
    def build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            value = np.argmax(np.bincount(y.astype(int)))
            return DecisionNode(value=value)

        feature_index, threshold = self.find_best_split(X, y)
        X_true, y_true, X_false, y_false = self.split_data(X, y, feature_index, threshold)

        true_branch = self.build_tree(X_true, y_true, depth + 1)
        false_branch = self.build_tree(X_false, y_false, depth + 1)

        return DecisionNode(feature_index=feature_index, threshold=threshold, true_branch=true_branch, false_branch=false_branch)

    # 训练决策树模型
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    # 预测单个样本
    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(x, node.true_branch)
        else:
            return self.predict_sample(x, node.false_branch)

    # 预测数据集
    def predict(self, X):
        return [self.predict_sample(x, self.tree) for x in X]

X,y=make_classification(n_samples=1000,#1000个样本
                        n_features=2,#两个特征，方便画图
                        n_informative=2,#信息特征(有用特征)
                        n_redundant=0,#冗余特征，它是信息特征的线性组合
                        n_repeated=0,#重复特征
                        n_classes=2,#分类样别
                        random_state=None,
                        n_clusters_per_class=2,#每个类别两簇
                        shuffle=True,
                        class_sep=1,#将每个簇分隔开来，较大的值将使分类任务更加容易
                        shift=10,
                        scale=3,
                        flip_y=0,)#没有噪声

# 创建并训练决策树模型
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)
# 可视化并保存决策树
# dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'],
#                            filled=True, rounded=True, special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('decision_tree.png')

# 预测数据集
y_pred = clf.predict(X)

# 绘制分类效果散点图
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Tree Classification')
plt.show()