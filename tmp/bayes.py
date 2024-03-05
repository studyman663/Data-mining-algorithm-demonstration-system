import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 定义贝叶斯分类器类
class NaiveBayesClassifier:
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

# 生成示例数据集
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 创建并训练贝叶斯分类器
model = NaiveBayesClassifier()
model.fit(X, y)

# 预测数据集
y_pred = model.predict(X)

# 绘制分类效果散点图
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='cool')
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Naive Bayes Classification')
plt.show()