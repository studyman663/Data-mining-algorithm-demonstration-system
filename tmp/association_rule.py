import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

# 创建示例数据集
dataset = [['Apple', 'Beer', 'Rice'],
           ['Apple', 'Beer'],
           ['Apple', 'Bananas'],
           ['Milk', 'Beer'],
           ['Milk', 'Rice'],
           ['Apple', 'Beer', 'Rice', 'Bananas'],
           ['Rice', 'Beer'],
           ['Rice', 'Bananas'],
           ['Apple', 'Rice', 'Bananas'],
           ['Apple', 'Bananas']]

# 将数据集进行one-hot编码
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 使用Apriori算法查找频繁项集
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# 根据频繁项集生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# 可视化关联规则
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules')
plt.show()