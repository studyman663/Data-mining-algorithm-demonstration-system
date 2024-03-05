import random

from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QComboBox, QTabWidget, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification

from fpModel import myApriori, myFPgrowth


class FPWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle('数据挖掘算法演示系统')
        self.setGeometry(400, 100, 800, 700)
        vbox = self.init_UI()
        self.setLayout(vbox)

    def init_UI(self):
        self.data = None
        self.mode = '模式数据'
        self.num = 10
        self.classes = 5
        mode = ['数据集生成方法', "模式数据", "随机数据"]
        num = ['事务总数', '5', '10','15','20','30']
        classes = ['商品类别数', '3', '4', '5','6','7']
        mode_combbox = QComboBox(self)
        mode_combbox.addItems(mode)
        num_combbox = QComboBox(self)
        num_combbox.addItems(num)
        classes_combbox = QComboBox(self)
        classes_combbox.addItems(classes)
        show_button = QPushButton('生成数据')

        tab_widget = QTabWidget()
        self.source_data = QLabel()
        self.source_data.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap = QPixmap("myplot.png")  # 替换为你的图片路径
        self.source_data.setPixmap(self.pixmap)
        self.apriori = FPTabWidget()
        self.fpgrowth = FPTabWidget()
        # self.bayes.depth_combbox.setVisible(False)
        self.apriori.run_button.clicked.connect(self.runApriori)
        self.fpgrowth.run_button.clicked.connect(self.runFPgrowth)

        tab_widget.addTab(self.source_data, '原始数据')
        tab_widget.addTab(self.apriori, 'Apriori')
        tab_widget.addTab(self.fpgrowth, 'FPGrowth')

        # 使用水平布局管理器布局lab_acc控件和account控件，左右留白10像素
        hbox_func = QHBoxLayout()  # 水平布局管理器
        hbox_func.addSpacing(10)  # 水平布局间隔
        hbox_func.addWidget(mode_combbox)  # 将实例化标签账号放入
        hbox_func.addWidget(classes_combbox)
        hbox_func.addWidget(num_combbox)  # 将单行文本编辑框放入
        hbox_func.addWidget(show_button)
        hbox_func.addSpacing(10)

        # 使用水平布局管理器布局lab_pw控件和passwd控件，左右留白10像素
        hbox_tab = QHBoxLayout()  # 垂直布局管理器
        hbox_tab.addWidget(tab_widget)

        # 使用垂直布局管理器布局上面3个水平布局管理器
        vbox = QVBoxLayout()

        vbox.addSpacing(10)
        vbox.addLayout(hbox_func)
        vbox.addLayout(hbox_tab)
        vbox.addSpacing(10)

        mode_combbox.currentTextChanged.connect(lambda: (self.modeSet(mode_combbox.currentText())))
        num_combbox.currentTextChanged.connect(lambda: (self.numSet(num_combbox.currentText())))
        classes_combbox.currentTextChanged.connect(lambda: (self.classesSet(classes_combbox.currentText())))
        show_button.clicked.connect(self.showData)
        return vbox

    def modeSet(self, mode):
        if mode == '数据集生成方法':
            QMessageBox.information(self, '提醒', '请选择数据集生成方法')
        else:
            self.mode = mode

    def numSet(self, num):
        if num == '事务总数':
            QMessageBox.information(self, '提醒', '请选择事务总数')
        else:
            self.num = int(num)

    def classesSet(self, classes):
        if classes == '商品类别数':
            QMessageBox.information(self, '提醒', '请选择商品类别数')
        else:
            self.classes = int(classes)

    def dataCreate(self):
        if self.mode == '随机数据':
            # 随机生成商品种类名
            categories = [f'商品{i}' for i in range(1, self.classes + 1)]
            dataset = []
            transaction_set = set()
            while len(dataset) < self.num:
                num_items = random.randint(1, self.classes)
                transaction = random.sample(categories, num_items)

                # 将事务转换为元组并判断是否已生成过
                transaction_tuple = tuple(sorted(transaction))
                if transaction_tuple not in transaction_set:
                    dataset.append(transaction)
                    transaction_set.add(transaction_tuple)
            return dataset
        else:
            return [
                ['A', 'B', 'C', 'E', 'F', 'O'],
                ['A', 'C', 'G'],
                ['E', 'I'],
                ['A', 'C', 'D', 'E', 'G'],
                ['A', 'C', 'E', 'G', 'L'],
                ['E', 'J'],
                ['A', 'B', 'C', 'E', 'F', 'P'],
                ['A', 'C', 'D'],
                ['A', 'C', 'E', 'G', 'M'],
                ['A', 'C', 'E', 'G', 'N'],
                ['A', 'C', 'B'],
                ['A', 'B', 'D']]

    def showData(self):
        self.data = self.dataCreate()
        print(self.data)
        # img_path = 'result/fp.png'
        # plt.scatter(self.data[:, 0], self.data[:, 1],c=self.label, cmap='cool')
        # plt.savefig(img_path)
        # pixmap = QPixmap(img_path)
        # self.source_data.setPixmap(pixmap)
        # self.decision_tree.image_label.setPixmap(pixmap)
        # self.bayes.image_label.setPixmap(pixmap)

    def runApriori(self):
        model = myApriori(self.data, self.apriori.minSup * len(self.data))
        img_path = model.Run()
        pixmap = QPixmap(img_path)
        self.apriori.image_label.setPixmap(pixmap)

    def runFPgrowth(self):
        model = myFPgrowth(self.data, self.fpgrowth.minSup * len(self.data))
        img_path = model.Run()
        pixmap = QPixmap(img_path)
        self.fpgrowth.image_label.setPixmap(pixmap)

    def closeEvent(self, evt):
        self.main_window.show()


class FPTabWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.minSup = 0.5
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap = QPixmap("myplot.png")  # 替换为你的图片路径
        self.image_label.setPixmap(self.pixmap)
        minSup = ['最小支持度', '0.1', '0.2', '0.3', '0.4', '0.5']
        self.minSup_combbox = QComboBox()
        self.minSup_combbox.addItems(minSup)

        self.minSup_combbox.currentTextChanged.connect(lambda: (self.minSupSet(self.minSup_combbox.currentText())))
        self.run_button = QPushButton('运行')
        self.result_label = QLabel()
        self.result_label.setText('性能分析')

        hbox1 = QHBoxLayout()

        hbox1.addSpacing(10)
        hbox1.addWidget(self.image_label)
        hbox1.addWidget(self.minSup_combbox)
        hbox1.addWidget(self.run_button)
        hbox1.addSpacing(10)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.result_label)

        vbox = QVBoxLayout()

        vbox.addSpacing(10)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addSpacing(10)

        self.setLayout(vbox)

    def minSupSet(self, minSup):
        if minSup == '最小支持度':
            QMessageBox.information(self, '提醒', '请选择最小支持度')
        else:
            self.minSup = float(minSup)
