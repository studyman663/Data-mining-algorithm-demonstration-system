import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QGroupBox, QHBoxLayout
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class VisualizationApp(QMainWindow):
    def __init__(self):
        super(VisualizationApp, self).__init__()

        # 创建页面
        self.setWindowTitle("可视化应用")
        # 获取屏幕大小
        screen = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen.width(), screen.height()
        # 计算窗口应该出现在屏幕正中央的位置
        window_width, window_height = 900, 600
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        # 设置窗口的位置和大小
        self.setGeometry(x, y, window_width, window_height)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        # 左侧 groupBox
        self.label_box = QGroupBox("可视化", self)
        self.label_box.setFixedWidth(150)  # 固定宽度
        self.layout.addWidget(self.label_box)

        # 垂直布局
        self.group_layout = QVBoxLayout(self.label_box)
        self.group_layout.setAlignment(Qt.AlignTop)  # 顶格分布

        # 饼图按钮
        self.pie_button = QPushButton("饼图", self)
        self.pie_button.clicked.connect(self.show_pie_chart)
        self.group_layout.addWidget(self.pie_button)

        # 柱状图按钮
        self.bar_button = QPushButton("柱状图", self)
        self.bar_button.clicked.connect(self.show_bar_chart)
        self.group_layout.addWidget(self.bar_button)

        # Matplotlib 白板
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)


    def show_pie_chart(self):
        data = [30, 40, 20, 10]  # 示例数据
        labels = ['A', 'B', 'C', 'D']  # 示例标签
        self.ax.clear()
        self.ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
        self.ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        self.canvas.draw()


    def show_bar_chart(self):
        data = [10, 30, 50, 20]  # 示例数据
        categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']  # 示例类别
        self.ax.clear()
        self.ax.bar(categories, data)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualizationApp()
    window.show()
    sys.exit(app.exec_())
