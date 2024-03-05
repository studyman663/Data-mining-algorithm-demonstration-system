import sys

import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QComboBox, \
    QTabWidget, QMessageBox
from PyQt6.QtGui import QIcon, QPixmap, QFont, QPalette, QBrush, QColor
from PyQt6.QtCore import Qt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.datasets import make_moons, make_blobs, make_circles

from ClassifyWindow import ClassifyWindow
from ClusterWindow import ClusterWindow
from FPWindow import FPWindow
from clusterModel import myKmeans, myKmedoid, myAgnes, myDbscan, myDiana


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # font = QFont("Serif", 60)
        # font.setBold(True)
        # self.setFont(font)
        # self.setStyleSheet("color: white;")
        self.setWindowTitle('数据挖掘算法演示系统')
        self.setGeometry(400, 100, 800, 700)
        self.title_label = QLabel(self)
        background_img = QPixmap("source/background.png")
        background_brush = QBrush(background_img)
        palette = QPalette()
        palette.setBrush(self.backgroundRole(), background_brush)
        self.setPalette(palette)
        vbox = self.init_UI()
        self.setLayout(vbox)
        self.show()  # 显示窗口

    def init_UI(self):
        self.fp_button = QPushButton('频繁模式挖掘')
        self.classify_button = QPushButton('分类')
        self.cluster_button = QPushButton('聚类')
        font_title = QFont("Serif", 40)
        font_title.setBold(True)
        font_button = QFont("Serif", 25)
        font_button.setBold(True)
        self.fp_button.clicked.connect(self.show_fp_window)
        self.classify_button.clicked.connect(self.show_classify_window)
        self.cluster_button.clicked.connect(self.show_cluster_window)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setText('数据挖掘算法演示系统')
        self.title_label.setStyleSheet("color: white;")
        self.title_label.setFont(font_title)
        self.fp_button.setFont(font_button)
        self.classify_button.setFont(font_button)
        self.cluster_button.setFont(font_button)
        self.fp_button.setFixedSize(250, 100)
        self.classify_button.setFixedSize(250, 100)
        self.cluster_button.setFixedSize(250, 100)
        # 使用水平布局管理器布局lab_acc控件和account控件，左右留白10像素
        hbox_title = QHBoxLayout()  # 水平布局管理器
        hbox_title.addWidget(self.title_label)
        hbox_fp = QHBoxLayout()
        hbox_fp.addWidget(self.fp_button)
        hbox_classify = QHBoxLayout()
        hbox_classify.addWidget(self.classify_button)
        hbox_cluster = QHBoxLayout()
        hbox_cluster.addWidget(self.cluster_button)
        # 使用垂直布局管理器布局上面3个水平布局管理器
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox_title)
        vbox.addStretch(1)
        vbox.addLayout(hbox_fp)
        vbox.addStretch(1)
        vbox.addLayout(hbox_classify)
        vbox.addStretch(1)
        vbox.addLayout(hbox_cluster)
        vbox.addStretch(1)
        return vbox

    def show_fp_window(self):
        self.hide()
        self.fp_window = FPWindow(self)
        self.fp_window.show()

    def show_classify_window(self):
        self.hide()
        self.classify_window = ClassifyWindow(self)
        self.classify_window.show()

    def show_cluster_window(self):
        self.hide()
        self.cluster_window = ClusterWindow(self)
        self.cluster_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec())
