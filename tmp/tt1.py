from matplotlib import pyplot as plt
from collections import OrderedDict

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
        print(freqItemList)
        support = []  # 存储支持度
        itemsets = []  # 存储频繁项集

        for i in range(len(freqItemList)):
            for j in range(len(freqItemList[i])):
                # print(freqItemList[i])
                itemsets.append(freqItemList[i][j][0])
                support.append(freqItemList[i][j][1])
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
        img_path='result/fpgrowth.png'
        plt.savefig(img_path)
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

def loadSimpDat():
    simpData = [['l1', 'l2', 'l5'],
            ['l2', 'l4'],
            ['l2', 'l3'],
            ['l1', 'l2', 'l4'],
            ['l1', 'l3'],
            ['l2', 'l3'],
            ['l1', 'l3'],
            ['l1', 'l2', 'l3', 'l5'],
            ['l1', 'l2', 'l3']]
    return simpData

dataset=loadSimpDat()
model=myFPgrowth(dataset,2)
model.Run()