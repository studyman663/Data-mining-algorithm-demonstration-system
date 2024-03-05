from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QComboBox, QTabWidget, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles

from clusterModel import myKmeans, myKmedoid, myDiana, myAgnes, myDbscan
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

class ClusterWindow(QWidget):
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
        self.type = '散点图'
        self.num = 300
        self.center = 4
        self.noise = 0
        mode = ['数据集生成方法', "模式数据", "随机数据"]
        type = ['数据集种类', "散点图", "漩涡图", "环状图"]
        num = ['样本总数', '100', '500', '1000']
        center = ['样本类别数', '1', '2', '3', '4', '5']
        noise = ['噪声大小', '0', '0.1', '0.2', '0.3']
        mode_combbox = QComboBox(self)
        mode_combbox.addItems(mode)
        type_combbox = QComboBox(self)
        type_combbox.addItems(type)
        num_combbox = QComboBox(self)
        num_combbox.addItems(num)
        center_combbox = QComboBox(self)
        center_combbox.addItems(center)
        noise_combbox = QComboBox(self)
        noise_combbox.addItems(noise)
        show_button = QPushButton('生成数据')

        tab_widget = QTabWidget()
        self.source_data = QLabel()
        self.source_data.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap = QPixmap("myplot.png")  # 替换为你的图片路径
        self.source_data.setPixmap(self.pixmap)
        self.kmeans = ClusterTabWidget()
        self.kmedoid = ClusterTabWidget()
        self.agnes = ClusterTabWidget()
        self.diana = ClusterTabWidget()
        self.dbscan = ClusterTabWidget()
        self.kmeans.min_combbox.setVisible(False)
        self.kmedoid.min_combbox.setVisible(False)
        self.agnes.min_combbox.setVisible(False)
        self.diana.min_combbox.setVisible(False)
        self.kmeans.radius_combbox.setVisible(False)
        self.kmedoid.radius_combbox.setVisible(False)
        self.agnes.radius_combbox.setVisible(False)
        self.diana.radius_combbox.setVisible(False)
        self.dbscan.center_combbox.setVisible(False)
        self.kmeans.run_button.clicked.connect(self.runKmeans)
        self.kmedoid.run_button.clicked.connect(self.runKmedoid)
        self.agnes.run_button.clicked.connect(self.runAgnes)
        self.diana.run_button.clicked.connect(self.runDiana)
        self.dbscan.run_button.clicked.connect(self.runDbscan)

        tab_widget.addTab(self.source_data, '原始数据')
        tab_widget.addTab(self.kmeans, 'kmeans')
        tab_widget.addTab(self.kmedoid, 'kmedoid')
        tab_widget.addTab(self.agnes, 'agnes')
        tab_widget.addTab(self.diana, 'diana')
        tab_widget.addTab(self.dbscan, 'dbscan')


        # 使用水平布局管理器布局lab_acc控件和account控件，左右留白10像素
        hbox_func = QHBoxLayout()  # 水平布局管理器
        hbox_func.addSpacing(10)  # 水平布局间隔
        hbox_func.addWidget(mode_combbox)  # 将实例化标签账号放入
        hbox_func.addWidget(type_combbox)  # 将单行文本编辑框放入
        hbox_func.addWidget(num_combbox)  # 将单行文本编辑框放入
        hbox_func.addWidget(center_combbox)
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
        type_combbox.currentTextChanged.connect(lambda: (self.typeSet(type_combbox.currentText())))
        num_combbox.currentTextChanged.connect(lambda: (self.numSet(num_combbox.currentText())))
        center_combbox.currentTextChanged.connect(lambda: (self.centerSet(center_combbox.currentText())))
        noise_combbox.currentTextChanged.connect(lambda: (self.noiseSet(noise_combbox.currentText())))
        show_button.clicked.connect(self.showData)
        return vbox

    def modeSet(self, mode):
        if mode == '数据集生成方法':
            QMessageBox.information(self, '提醒', '请选择数据集生成方法')
        else:
            self.mode = mode

    def typeSet(self, type):
        if type == '数据集种类':
            QMessageBox.information(self, '提醒', '请选择数据集种类')
        else:
            self.type = type

    def numSet(self, num):
        if num == '样本总数':
            QMessageBox.information(self, '提醒', '请选择样本总数')
        else:
            self.num = int(num)

    def centerSet(self, center):
        if center == '样本类别数':
            QMessageBox.information(self, '提醒', '请选择样本类别数')
        else:
            self.center = int(center)

    def noiseSet(self, noise):
        if noise == '噪声大小':
            QMessageBox.information(self, '提醒', '请选择噪声大小')
        else:
            self.noise = float(noise)

    def dataCreate(self):
        data, label = None, None
        if self.mode == '模式数据':
            if self.type == '散点图':
                data, label = make_blobs(n_samples=self.num, centers=self.center, random_state=42)
            elif self.type == '漩涡图':
                data, label = make_moons(n_samples=self.num, noise=self.noise, random_state=42)
            else:
                data, label = make_circles(n_samples=self.num, noise=self.noise, random_state=42)
        else:
            pass
        return data, label

    def showData(self):
        self.data, self.label = self.dataCreate()
        img_path = 'result/cluster.png'
        plt.scatter(self.data[:, 0], self.data[:, 1])
        plt.savefig(img_path)
        plt.close()
        pixmap = QPixmap(img_path)
        self.source_data.setPixmap(pixmap)
        self.kmeans.image_label.setPixmap(pixmap)
        self.kmedoid.image_label.setPixmap(pixmap)
        self.diana.image_label.setPixmap(pixmap)
        self.agnes.image_label.setPixmap(pixmap)
        self.dbscan.image_label.setPixmap(pixmap)

    def runKmeans(self):
        model = myKmeans()
        img_path = model.Run(self.data, self.kmeans.center)
        pixmap = QPixmap(img_path)
        self.kmeans.image_label.setPixmap(pixmap)

    def runKmedoid(self):
        model = myKmedoid()
        img_path = model.Run(self.data, self.kmedoid.center)
        pixmap = QPixmap(img_path)
        self.kmedoid.image_label.setPixmap(pixmap)

    def runDiana(self):
        model = myDiana()
        img_path = model.Run(self.data, 4)
        pixmap = QPixmap(img_path)
        self.diana.image_label.setPixmap(pixmap)

    def runAgnes(self):
        print(self.data,self.agnes.center)
        model = myAgnes()
        img_path = model.Run(self.data, 4)
        pixmap = QPixmap(img_path)
        self.agnes.image_label.setPixmap(pixmap)

    def runDbscan(self):
        model = myDbscan(eps=self.dbscan.radius, min_samples=self.dbscan.min)
        clusters = model.Run(self.data)
        img_path=model.visualize_clusters(self.data, clusters)
        pixmap = QPixmap(img_path)
        self.dbscan.image_label.setPixmap(pixmap)

    def closeEvent(self, evt):
        self.main_window.show()


class ClusterTabWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.center = 2
        self.min=5
        self.radius=1
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap = QPixmap("myplot.png")  # 替换为你的图片路径
        self.image_label.setPixmap(self.pixmap)
        radius = ['领域半径', '0.1', '0.5', '1', '2', '5']
        self.radius_combbox = QComboBox()
        self.radius_combbox.addItems(radius)
        min = ['最小样本数', '1', '2', '3', '4', '5']
        self.min_combbox = QComboBox()
        self.min_combbox.addItems(min)
        center = ['样本类别数', '1', '2', '3', '4', '5']
        self.center_combbox = QComboBox()
        self.center_combbox.addItems(center)

        self.radius_combbox.currentTextChanged.connect(lambda: (self.radiusSet(self.radius_combbox.currentText())))
        self.min_combbox.currentTextChanged.connect(lambda: (self.minSet(self.min_combbox.currentText())))
        self.center_combbox.currentTextChanged.connect(lambda: (self.centerSet(self.center_combbox.currentText())))
        self.run_button = QPushButton('运行')
        self.result_label = QLabel()
        self.result_label.setText('性能分析')

        hbox1 = QHBoxLayout()

        hbox1.addSpacing(10)
        hbox1.addWidget(self.image_label)
        hbox1.addWidget(self.radius_combbox)
        hbox1.addWidget(self.min_combbox)
        hbox1.addWidget(self.center_combbox)
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

    def centerSet(self, center):
        if center == '样本类别数':
            QMessageBox.information(self, '提醒', '请选择样本类别数')
        else:
            self.center = int(center)

    def minSet(self, min):
        if min == '最小样本数':
            QMessageBox.information(self, '提醒', '请选择最小样本数')
        else:
            self.min = int(min)

    def radiusSet(self, radius):
        if radius == '领域半径':
            QMessageBox.information(self, '提醒', '请选择领域半径')
        else:
            self.radius = float(radius)