from collections import OrderedDict
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

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
        fig, ax = plt.subplots(figsize=(6.4, 4.8))  # 设置图形大小

        ax.barh(range(len(sorted_itemsets)), sorted_support, align='center')
        ax.set_yticks(range(len(sorted_itemsets)))
        ax.set_yticklabels([str(itemset) for itemset in sorted_itemsets], fontsize=8)  # 调整纵坐标轴标签字体大小
        ax.set_xlabel('支持度')
        ax.set_ylabel('项集')
        ax.set_title('频繁项集')

        plt.tight_layout()  # 调整图形布局，避免标签被遮挡
        img_path= 'result/apriori.png'
        plt.savefig(img_path)
        plt.close()
        return img_path

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

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # 节点元素名称
        self.count = numOccur  # 节点出现的次数
        self.nodeLink = None  # 指向下一个相似节点
        self.parent = parentNode  # 指向父节点
        self.children = {}  # 指向子节点，子节点元素名称为键，指向子节点指针为值

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        for child in self.children.values():
            child.disp(ind + 1)  # 打印时，子节点的缩进比父节点更深一级
class myFPgrowth():
    def __init__(self,dataset,minSup):
        self.dataset=dataset
        self.minSup=minSup
    def Run(self):
        initSet = self.createInitSet()
        myFPtree, myHeaderTab = self.createTree(initSet)
        myFPtree.disp()
        freqItemList = []
        self.mineTree(myFPtree, myHeaderTab, set([]), freqItemList)
        support = []  # 存储支持度
        itemsets = []  # 存储频繁项集

        for i in range(len(freqItemList)):
            for j in range(len(freqItemList[i])):
                itemset = freqItemList[i][j][0]
                count = freqItemList[i][j][1] / len(initSet)
                itemsets.append(itemset)
                support.append(count)
        # 对频繁项集和支持度按照支持度进行排序
        sorted_itemsets = [x for _, x in sorted(zip(support, itemsets), reverse=False)]
        sorted_support = sorted(support, reverse=False)

        # 创建水平柱状图
        fig, ax = plt.subplots(figsize=(6.4, 4.8))  # 设置图形大小

        ax.barh(range(len(sorted_itemsets)), sorted_support, align='center')
        ax.set_yticks(range(len(sorted_itemsets)))
        ax.set_yticklabels([str(itemset) for itemset in sorted_itemsets], fontsize=8)  # 调整纵坐标轴标签字体大小
        ax.set_xlabel('支持度')
        ax.set_ylabel('项集')
        ax.set_title('频繁项集')

        plt.tight_layout()  # 调整图形布局，避免标签被遮挡
        img_path= 'result/fpgrowth.png'
        plt.savefig(img_path)
        plt.close()
        return img_path

    def createInitSet(self):
        retDict = OrderedDict()  # retDict = {}
        for trans in self.dataset:
            retDict[frozenset(trans)] = 1
        return retDict

    def createTree(self,dataset):
        headerTable = {}  # 支持度>=minSup的dist{所有元素: 出现的次数}
        for trans in dataset:  # 循环 dist{行: 出现次数}的样本数据
            for item in trans:
                headerTable[item] = headerTable.get(item, 0) + dataset[trans]
        for k in list(headerTable.keys()):  # python3中.keys()返回的是迭代器不是list,不能在遍历时对其改变。
            if headerTable[k] < self.minSup:
                del (headerTable[k])  # 删除不满足最小支持度的元素
        freqItemSet = set(headerTable.keys())  # 满足最小支持度的频繁项集 # 满足minSup: set(各元素集合)
        if len(freqItemSet) == 0:  # 如果不存在，直接返回None
            return None, None
        for k in headerTable:  # 我们在每个键对应的值中增加一个“None”，为后面的存储相似元素做准备
            headerTable[k] = [headerTable[k], None]  # 格式化:  dist{元素key: [元素次数, None]}
        retTree = treeNode('Null Set', 1, None)  # create tree
        for tranSet, count in dataset.items():  # 循环 dist{行: 出现次数}的样本数据
            localD = {}  # localD = dist{元素key: 元素总出现次数}
            for item in tranSet:
                if item in freqItemSet:  # 过滤，只取该样本中满足最小支持度的频繁项
                    localD[item] = headerTable[item][0]
            if len(localD) > 0:  # 如果该条记录有符合条件的元素
                orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
                self.updateTree(orderedItems, retTree, headerTable, count)
        return retTree, headerTable

    def updateTree(self,items, inTree, headerTable, count):
        if items[0] in inTree.children:  # 如果inTree的子节点中已经存在该元素
            inTree.children[items[0]].inc(count)
        else:
            inTree.children[items[0]] = treeNode(items[0], count, inTree)
            if headerTable[items[0]][1] is None:  # 如果在相似元素的字典headerTable中，该元素键对应的列表值中，起始元素为None
                headerTable[items[0]][1] = inTree.children[items[0]]  # 把新创建的这个节点赋值给起始元素
            else:
                self.updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
        if len(items) > 1:
            self.updateTree(items[1::], inTree.children[items[0]], headerTable, count)

    def updateHeader(self,nodeToTest, targetNode):
        while nodeToTest.nodeLink is not None:
            nodeToTest = nodeToTest.nodeLink
        nodeToTest.nodeLink = targetNode

    def ascendTree(self,leafNode, prefixPath):
        if leafNode.parent is not None:
            prefixPath.append(leafNode.name)
            self.ascendTree(leafNode.parent, prefixPath)

    def findPrefixPath(self,basePat, treeNode):
        condPats = {}
        while treeNode is not None:  # 对 treeNode的link进行循环
            prefixPath = []
            self.ascendTree(treeNode, prefixPath)  # 寻找改节点的父节点，相当于找到了该节点的频繁项集
            if len(prefixPath) > 1:  # 排除自身这个元素，判断是否存在父元素（所以要>1, 说明存在父元素）
                condPats[frozenset(prefixPath[1:])] = treeNode.count
            treeNode = treeNode.nodeLink
        return condPats

    def mineTree(self,inTree, headerTable, preFix, freqItemList):
        # 对头表中的项按照支持度降序排序
        sortedHeader = sorted(headerTable.items(), key=lambda p: p[1][0])

        # 从底部开始，逐个获取频繁项集
        for basePat, basePatCount in sortedHeader:
            # 构建当前项的频繁项集
            newFreqSet = preFix.copy()
            newFreqSet.add(basePat)
            # freqItemList.append(newFreqSet)
            # 获取k
            k = len(newFreqSet)

            # 扩容k项集列表
            if k > len(freqItemList):
                freqItemList.append([])

            # 添加到对应k项集列表
            freqItemList[k - 1].append([list(newFreqSet), basePatCount[0]])

            # 生成当前项的条件模式基
            condPattBases = self.findPrefixPath(basePat, headerTable[basePat][1])
            # 构建当前项的条件FP树
            myCondTree, myHead = self.createTree(condPattBases)

            # 递归挖掘条件FP树
            if myHead is not None:
                self.mineTree(myCondTree, myHead, newFreqSet, freqItemList)