import numpy as np

class myBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.num_features = X.shape[1]
        self.class_priors = np.zeros(self.num_classes)
        self.feature_likelihoods = []

        # 计算每个类别的先验概率
        for i, c in enumerate(self.classes):
            class_mask = (y == c)
            self.class_priors[i] = np.sum(class_mask) / len(y)

            # 计算每个特征在给定类别下的似然概率
            feature_likelihood = []
            for j in range(self.num_features):
                feature_values = np.unique(X[:, j])
                feature_prob = []
                for value in feature_values:
                    feature_mask = (X[:, j] == value)
                    feature_class_mask = np.logical_and(feature_mask, class_mask)
                    prob = np.sum(feature_class_mask) / np.sum(class_mask)
                    feature_prob.append(prob)
                feature_likelihood.append(feature_prob)
            self.feature_likelihoods.append(feature_likelihood)

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = []
            for i, c in enumerate(self.classes):
                class_score = np.log(self.class_priors[i])
                for j in range(self.num_features):
                    feature_values = np.unique(X[:, j])
                    feature_likelihood = self.feature_likelihoods[i][j]
                    if x[j] in feature_values:
                        likelihood_index = np.argmax(feature_values == x[j])
                        class_score += np.log(feature_likelihood[likelihood_index])
                class_scores.append(class_score)
            predictions.append(self.classes[np.argmax(class_scores)])
        return predictions


class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_index = feature_index  # 特征索引
        self.threshold = threshold  # 分割阈值
        self.value = value  # 叶节点值
        self.true_branch = true_branch  # 左子树
        self.false_branch = false_branch  # 右子树

# 定义决策树分类器类
class myDecisionTree:
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