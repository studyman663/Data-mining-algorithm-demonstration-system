from PyQt6.QtWidgets import QWidget, QLabel

class CodeWindow(QWidget):
    def __init__(self,code):
        super().__init__()
        self.code = code
        self.label = QLabel(self.code,self)
        self.setWindowTitle('核心代码展示')
        self.setGeometry(700, 300, 350, 500)


