#import pandas as pd
# from mlxtend.frequent_patterns import apriori,fpgrowth,association_rules
#
# df=pd.DataFrame({
#     'apple':[1,0,1,1,1,0],
#     'orange':[0,1,0,1,1,1],
#     'mango':[1,1,0,1,0,1],
#     'banana':[0,1,1,1,0,0],
#     'blueberry':[1,1,1,1,0,1],
#     'cherry':[0,0,1,0,1,1]
# })
#
# frequent_itemsets=apriori(df=df,use_colnames=True)
# rules = association_rules(df=frequent_itemsets, metric='lift', min_threshold=1)
# print(rules[(rules['confidence']>0.8)&(rules['lift']>1.125)])
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori

# 输入数据集为二维数组
dataset = [
        ['A', 'B', 'C', 'E', 'F','O'],
        ['A', 'C', 'G'],
        ['E','I'],
        ['A', 'C', 'D', 'E', 'G'],
        ['A', 'C', 'E', 'G', 'L'],
        ['E', 'J'],
        ['A', 'B', 'C', 'E', 'F', 'P'],
        ['A', 'C', 'D'],
        ['A', 'C', 'E', 'G', 'M'],
        ['A', 'C', 'E', 'G', 'N'],
        ['A', 'C', 'B'],
        ['A', 'B', 'D']]

# 将数据集转换为适用于mlxtend的格式
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 使用FPGrowth算法进行频繁项集挖掘
frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)
frequent_itemsets1 = apriori(df, min_support=0.5, use_colnames=True)

# 打印频繁项集及其支持度
print('fp:')
print(frequent_itemsets)
print('apriori:')
print(frequent_itemsets1)