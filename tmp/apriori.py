from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
def load_data():
    data = [['g2', 'g1', 'g3', 'g5'], ['g4', 'g1', 'g3'], ['g2', 'g3', 'g4', 'g5', 'g1'], ['g3', 'g1'], ['g2', 'g5', 'g1'], ['g1', 'g2', 'g4'], ['g5'], ['g2'], ['g4', 'g3', 'g1', 'g2', 'g5'], ['g4', 'g5'], ['g1', 'g5', 'g3', 'g4', 'g2'], ['g1', 'g3', 'g4', 'g5'], ['g1', 'g2', 'g5', 'g4'], ['g1', 'g5', 'g3'], ['g3'], ['g3'], ['g3', 'g5', 'g1', 'g2'], ['g2', 'g3', 'g5'], ['g4', 'g2', 'g3', 'g1'], ['g4', 'g1', 'g5', 'g2']]
    return data

class myApriori:

    def __init__(self, data, min_support):
        self.data = data
        self.min_support = min_support
    def Run(self):
        L = {}  # 储存一项集
        for i in self.data:  # 得到所有的一项集
            for j in i:
                if j in L:
                    L[j] += 1
                else:
                    L[j] = 1
        L_1 = []  # 储存频繁一项集
        for i in L.keys():  # 取其中所有的频繁集
            if L[i] >= self.min_support:
                L_1.append([[i], L[i]])
        L = []  # 频繁1~4项集
        L.append(sorted(L_1, key=lambda x: x[0]))  # 按键值排序转为列表

        for i in range(3):
            L.append(self.apriori(L[i]))  # 求频繁2项集
        support = []  # 存储支持度
        itemsets = []  # 存储频繁项集

        for i in range(len(L)):
            for j in range(len(L[i])):
                itemset = L[i][j][0]
                count = L[i][j][1] / len(self.data)
                itemsets.append(itemset)
                support.append(count)

        # 对频繁项集和支持度按照支持度进行排序
        sorted_itemsets = [x for _, x in sorted(zip(support, itemsets), reverse=False)]
        sorted_support = sorted(support, reverse=False)

        # 创建水平柱状图
        fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形大小

        ax.barh(range(len(sorted_itemsets)), sorted_support, align='center')
        ax.set_yticks(range(len(sorted_itemsets)))
        ax.set_yticklabels([str(itemset) for itemset in sorted_itemsets], fontsize=8)  # 调整纵坐标轴标签字体大小
        ax.set_xlabel('支持度')
        ax.set_ylabel('项集')
        ax.set_title('频繁项集')

        plt.tight_layout()  # 调整图形布局，避免标签被遮挡

        plt.show()
        return L

    # 根据项目，寻找支持度计数
    def xunzhao(self,s, L):
        k = len(s) - 1  # 根据项目长度，决定在频繁几项集中寻找
        if k < 0:  # s为空的情况
            print(s, '不是有效输入')
        # 标志变量，如果s中的项目有和当前比较的不同的，就置0；
        t = 0
        for i in range(len(L[k])):  # 遍历频繁k项集
            t = 1
            for j in range(k + 1):  # 开始比较是不是该项目
                if L[k][i][0][j] != s[j]:
                    t = 0
                    break
            if t:
                return L[k][i][1]  # 是的话，返回该项目的支持度计数
        print('未找到')  # 遍历结束的话说明未找到
        return -1  # 返回 -1 (无)

    # 得到划分，并顺便求出置信度
    def huafen(self,L, l, conf):  # 划分得到置信度
        X = l[0]  # 需要划分的列表
        lx = len(X)  # 长度
        # 用二进制的性质求真子集
        for i in range(1, 2 ** lx - 1):
            s1 = []
            s2 = []
            for j in range(lx):
                # 二进制末尾是0就进s1,否则就进s2达到划分目的
                if (i >> j) % 2:
                    s1.append(X[j])
                else:
                    s2.append(X[j])
            conf[str(s1), '->', str(s2)] = l[1] / self.xunzhao(s1, L)

    # 判断s中的数据在data中的数目
    def jishu(self,s):
        c = 0  # 记录出现数目
        # 标志变量，如果s中的数据有在data这一行不存在的，就置0；
        t = 0
        for ii in self.data:  # 数据每一行
            t = 1
            for jj in s:  # 对于 s 中的每一个项
                if not (jj in ii):
                    t = 0
                    break
            c += t  # 如果 s 在这一行存在，c++
        return c

    # 输入频繁k-1项集，支持度计数，数据，输出频繁k项集
    def apriori(self,L):
        if not L:
            return L
        L_ = []  # 频繁k项集
        L = [i[0] for i in L]  # k-1项集包含哪些
        k = len(L[0]) + 1  # 几项集
        L_len = len(L)
        up = k - 2  # 拼接同项长度
        i = 0
        while i < L_len - 1:  # 只剩最后一个时肯定没法拼
            A = L[i][0:up]  # 拼接前项
            c = i  # 记录走到那个前项了
            i += 1  # i 到下一个
            for j in range(c + 1, L_len):
                if L[j][0:up] != A:  # 前几项不一致就停止
                    i = j  # i 快进到发现新键值的地方
                    break
                else:  # 前几项一致时
                    s = L[c] + L[j][up:]  # 生成预选项
                    t = self.jishu(s)  # 得到 s 的支持度计数
                    if t >= self.min_support:  # 支持度计数大于self.min_support
                        L_.append([s, t])  # 添加到频繁项集中
        return L_


data = load_data()  # 生成数据，存到data
model=myApriori(data,0.5*len(data))
l=model.Run()
print(l)