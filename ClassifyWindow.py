from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QComboBox, QTabWidget, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification

from classifyModel import myDecisionTree, myBayes
plt.rcParams['font.sans-serif'] = ['SimHei']

class ClassifyWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle('数据挖掘算法演示系统')
        self.setGeometry(400, 100, 800, 700)
        vbox = self.init_UI()
        self.setLayout(vbox)

    def init_UI(self):
        self.data=None
        self.label=None
        self.mode = '模式数据'
        self.num = 300
        self.feature = 2
        self.classes = 2
        self.noise=0.1
        mode = ['数据集生成方法', "模式数据", "随机数据"]
        num = ['样本总数', '100', '500', '1000']
        feature = ['样本特征数', '1', '2', '3', '4', '5']
        classes = ['样本类别数', '1', '2', '3', '4', '5']
        noise = ['噪声大小', '0', '0.1', '0.2', '0.3']
        mode_combbox = QComboBox(self)
        mode_combbox.addItems(mode)
        num_combbox = QComboBox(self)
        num_combbox.addItems(num)
        feature_combbox = QComboBox(self)
        feature_combbox.addItems(feature)
        classes_combbox = QComboBox(self)
        classes_combbox.addItems(classes)
        noise_combbox = QComboBox(self)
        noise_combbox.addItems(noise)
        show_button = QPushButton('生成数据')

        tab_widget = QTabWidget()
        self.source_data = QLabel()
        self.source_data.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap = QPixmap("myplot.png")  # 替换为你的图片路径
        self.source_data.setPixmap(self.pixmap)
        self.decision_tree = ClassifyTabWidget()
        self.bayes = ClassifyTabWidget()
        self.bayes.depth_combbox.setVisible(False)
        self.decision_tree.run_button.clicked.connect(self.runDtree)
        self.bayes.run_button.clicked.connect(self.runBayes)

        tab_widget.addTab(self.source_data, '原始数据')
        tab_widget.addTab(self.decision_tree, '决策树')
        tab_widget.addTab(self.bayes, '贝叶斯')


        # 使用水平布局管理器布局lab_acc控件和account控件，左右留白10像素
        hbox_func = QHBoxLayout()  # 水平布局管理器
        hbox_func.addSpacing(10)  # 水平布局间隔
        hbox_func.addWidget(mode_combbox)  # 将实例化标签账号放入
        hbox_func.addWidget(num_combbox)  # 将单行文本编辑框放入
        hbox_func.addWidget(feature_combbox)  # 将单行文本编辑框放入
        hbox_func.addWidget(classes_combbox)
        hbox_func.addWidget(noise_combbox)
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
        feature_combbox.currentTextChanged.connect(lambda: (self.featureSet(feature_combbox.currentText())))
        classes_combbox.currentTextChanged.connect(lambda: (self.classesSet(classes_combbox.currentText())))
        noise_combbox.currentTextChanged.connect(lambda: (self.noiseSet(noise_combbox.currentText())))
        show_button.clicked.connect(self.showData)
        return vbox

    def modeSet(self, mode):
        if mode == '数据集生成方法':
            QMessageBox.information(self, '提醒', '请选择数据集生成方法')
        else:
            self.mode = mode

    def featureSet(self, feature):
        if type == '样本特征数':
            QMessageBox.information(self, '提醒', '请选择样本特征数')
        else:
            self.feature = int(feature)

    def numSet(self, num):
        if num == '样本总数':
            QMessageBox.information(self, '提醒', '请选择样本总数')
        else:
            self.num = int(num)

    def classesSet(self, classes):
        if classes == '样本类别数':
            QMessageBox.information(self, '提醒', '请选择样本类别数')
        else:
            self.classes = int(classes)

    def noiseSet(self, noise):
        if noise == '噪声大小':
            QMessageBox.information(self, '提醒', '请选择噪声大小')
        else:
            self.noise = float(noise)

    def dataCreate(self):
        data, label = None, None
        if self.mode == '模式数据':
            data, label =make_classification(n_samples=self.num,
                        n_features=self.feature,
                        n_informative=2,
                        n_redundant=0,
                        n_repeated=0,
                        n_classes=self.classes,
                        random_state=42,
                        n_clusters_per_class=2,
                        shuffle=True,
                        class_sep=1,
                        shift=10,
                        scale=3,
                        flip_y=self.noise)
        else:
            pass
        return data, label

    def showData(self):
        self.data, self.label = self.dataCreate()
        img_path = 'result/classify.png'
        plt.scatter(self.data[:, 0], self.data[:, 1],c=self.label, cmap='cool')
        plt.savefig(img_path)
        pixmap = QPixmap(img_path)
        self.source_data.setPixmap(pixmap)
        self.decision_tree.image_label.setPixmap(pixmap)
        self.bayes.image_label.setPixmap(pixmap)

    def runDtree(self):
        model = myDecisionTree(max_depth=self.decision_tree.depth)
        model.fit(self.data,self.label)
        pred=model.predict(self.data)
        plt.scatter(self.data[:, 0], self.data[:, 1], c=pred, cmap='bwr')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        img_path = 'result/decision_tree.png'
        plt.savefig(img_path)
        plt.close()
        pixmap = QPixmap(img_path)
        self.decision_tree.image_label.setPixmap(pixmap)

    def runBayes(self):
        model = myBayes()
        model.fit(self.data, self.label)
        pred = model.predict(self.data)
        plt.scatter(self.data[:, 0], self.data[:, 1], c=pred, cmap='bwr')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        img_path = 'result/bayes.png'
        plt.savefig(img_path)
        plt.close()
        pixmap = QPixmap(img_path)
        self.bayes.image_label.setPixmap(pixmap)


    def closeEvent(self, evt):
        self.main_window.show()

class ClassifyTabWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.depth = 5
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap = QPixmap("myplot.png")  # 替换为你的图片路径
        self.image_label.setPixmap(self.pixmap)
        depth = ['最大深度', '1', '2', '3', '4', '5']
        self.depth_combbox = QComboBox()
        self.depth_combbox.addItems(depth)

        self.depth_combbox.currentTextChanged.connect(lambda: (self.depthSet(self.depth_combbox.currentText())))
        self.run_button = QPushButton('运行')
        self.result_label = QLabel()
        self.result_label.setText('性能分析')

        hbox1 = QHBoxLayout()

        hbox1.addSpacing(10)
        hbox1.addWidget(self.image_label)
        hbox1.addWidget(self.depth_combbox)
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

    def depthSet(self, depth):
        if depth == '最大深度':
            QMessageBox.information(self, '提醒', '请选择最大深度')
        else:
            self.depth = int(depth)