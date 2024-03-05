'''
Author: Vici__
date: 2020/5/13
'''
import math

'''
Point类，记录坐标x，y和点的名字id
'''


class Point:
    '''
    初始化函数
    '''

    def __init__(self, x, y, name, id):
        self.x = x  # 横坐标
        self.y = y  # 纵坐标
        self.name = name  # 名字
        self.id = id  # 编号

    '''
    计算两点之间的欧几里得距离
    '''

    def calc_Euclidean_distance(self, p2):
        return math.sqrt((self.x - p2.x) * (self.x - p2.x) + (self.y - p2.y) * (self.y - p2.y))


'''
1. 获取数据集
'''


def get_dataset():
    # 原始数据集以元组形式存放，(横坐标，纵坐标，编号)
    datas = [(0, 2, 'A'), (0, 0, 'B'), (1.5, 0, 'C'), (5, 0, 'D'), (5, 2, 'E')]
    dataset = []  # 用于计算两点之间的距离，形式 [point1, point2...]
    id_point_dict = {}  # 编号和点的映射
    temp_list = []
    for i in range(len(datas)):  # 遍历原始数据集
        point = Point(datas[i][0], datas[i][1], datas[i][2], i)  # 利用(横坐标，纵坐标，编号)实例化
        id_point_dict[str(i)] = point
        dataset.append(point)  # 放入dataset中
        temp_list.append(point)
    return dataset, id_point_dict  # [p1, p2], {id: point}


'''
2. 计算任意两点之间的距离
'''


def get_dist(dataset):
    n = len(dataset)  # 点的个数
    dist = []  # 存放任意两点之间的距离
    for i in range(n):
        dist_i = []  # 临时列表
        for j in range(n):  # 遍历数据集
            # 计算距离并放入临时列表中
            dist_i.append(dataset[i].calc_Euclidean_distance(dataset[j]))
        dist.append(dist_i)  # 利用临时列表创建二维列表
    # 打印dist
    print("任意两点之间的距离：")
    for d in dist:
        print(d)
    print()
    return dist


'''
3. 计算簇内数据点相异度
'''


def get_dissimilitude(dist, ids):
    n = len(ids)  # 这个簇的数据点个数
    dissimilitudes = {}  # 存放数据点相异度
    for id1 in ids:
        id1_num = int(id1)
        d = 0  # 点id1的相异度，初始化为0
        for id2 in ids:  # 遍历其它数据点
            id2_num = int(id2)
            d += dist[id1_num][id2_num]  # 加上两点距离
        dissimilitudes[id1] = d / (n - 1)  # 计算相异度
    return dissimilitudes


'''
4. 寻找最大相异度的点
'''


def get_max_diff(dissimilitudes):
    Max = -1  # 最大相异度值，初始化为一个负值
    Max_id = -1  # 最大相异度值的数据点编号
    for id, diff in dissimilitudes.items():  # 遍历之前得到的相异度字典
        if diff > Max:  # 有更大的，就更新
            Max = diff
            Max_id = id
    return Max_id  # 返回最大相异度值的数据点编号


'''
5. DIANA算法主函数
'''


def DIANA(dataset, k, id_point_dict):
    dist = get_dist(dataset)  # 获取任意两点之间距离（欧几里得距离）
    res = []  # 结果列表，存放每次操作完成后的簇组合
    ids = []  # 初始簇
    for i in range(len(dataset)):
        ids.append(str(i))  # 初始簇中包含所有数据点的编号
    res.append(ids)  # 初始簇入结果列表

    while len(res) < k:  # 簇的个数为k个时，退出循环
        t_res = []  # 结果列表res的复制，只用于遍历
        for t in res:
            t_res.append(t)
        for ids in t_res:  # 遍历复制的结果列表
            splinter_group = []  # splinter group
            old_party = []  # old party
            dissimilitudes = get_dissimilitude(dist, ids)  # 计算ids这个簇的相异度
            Max_id = get_max_diff(dissimilitudes)  # 得到这个簇里最大相异度的数据点
            splinter_group.append(Max_id)  # 放入splinter group
            for id in ids:  # 其余数据点放入old party
                old_party.append(id)
            old_party.remove(Max_id)  # 全放进去，然后把最大点删掉就可以了
            pre_len = -1  # 用于判断old_party列表不再增加时，退出循环
            while pre_len != len(old_party):  # 不相等说明，old_party列表还在变化
                pre_len = len(old_party)  # 更新pre_len
                change_ids = []
                # 在old party中寻找 到splinter group中的点（E点）的最近距离
                # 小于等于到old party中的点的最近距离的点，找出D点，
                # 把该点加入splinter group中。在此数据集中，
                # 仅有点D到点E的距离2.3<3.5（5.3，5，3.5），
                # 所以将点D加入到splinter group 中（D,E点）；
                for id1 in old_party:  # 在old party中寻找，遍历
                    Min = float("INF")
                    flag = True  # 判断该点是否符合要求
                    for id2 in splinter_group:  # splinter_group中若有多个点，需要找到最近距离
                        if dist[int(id1)][int(id2)] < Min:
                            Min = dist[int(id1)][int(id2)]
                    for id3 in old_party:  # 寻找最近距离小于等于到old party中的点的最近距离的点
                        if (Min > dist[int(id1)][int(id3)]) and (id1 != id3):  # 不符合要求的置为False，并退出循环
                            flag = False
                            break
                    if flag:  # 该点符合要求
                        change_ids.append(id1)  # 放入change_ids列表中，表示需要变化的数据点
                for id in change_ids:  # 遍历
                    old_party.remove(id)  # 从old_party中删除
                    splinter_group.append(id)  # 放入splinter_group
            if len(splinter_group) != 0 and len(old_party) != 0:  # 当前簇发生变化了，更新结果列表res
                res.remove(ids)  # 删除旧簇
                res.append(splinter_group)  # 加入两个新簇
                res.append(old_party)
            # 打印结果看看
            print("-------------------------")
            print("最终聚类结果：")
            for r in res:
                for id in r:
                    # 我们之前用的都是数据点的编号，用id_point_dict找到该点，并打印他的名字
                    print(id_point_dict[id].name, end="")
                print()


# 测试
dataset, id_point_dict = get_dataset()
k = 2
DIANA(dataset, k, id_point_dict)