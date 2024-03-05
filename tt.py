import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.Qsci import QsciScintilla, QsciLexerPython

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Code Editor")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        editor = QsciScintilla()
        lexer = QsciLexerPython(editor)
        editor.setLexer(lexer)

        # 设置代码编辑器的样式
        editor.setMarginLineNumbers(1, True)
        editor.setMarginWidth(1, "00000")
        editor.setWrapMode(QsciScintilla.WrapMode.Word)

        layout.addWidget(editor)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())