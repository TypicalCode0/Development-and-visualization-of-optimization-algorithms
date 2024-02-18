from PyQt6.QtWidgets import QApplication, QMainWindow, QSplitter, QWidget, QLineEdit, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 1200, 800)  # x, y, width, height
        self.setWindowTitle('course_work')

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: black }")

        self.widget1 = QWidget(self.splitter)
        self.widget1.setStyleSheet("background-color: white; border: 2px solid black;")
        self.layout1 = QVBoxLayout(self.widget1)
        self.layout1.setContentsMargins(10, 10, 10, 10)
        self.layout1.setSpacing(0)

        self.label = QLabel("Enter the function:", self.widget1)
        self.label.setStyleSheet("border: none;")
        self.layout1.addWidget(self.label)

        self.input_line = QLineEdit(self.widget1)
        self.input_line.setStyleSheet("border: none;")
        self.input_line.returnPressed.connect(self.on_return_pressed)
        self.layout1.addWidget(self.input_line)
        self.layout1.addStretch()
        self.input_line.setStyleSheet("background-color: white; border: 1px solid black;")

        self.widget2 = QWidget(self.splitter)
        self.widget2.setStyleSheet("background-color: grey; border: 2px solid black;")

        self.splitter.setSizes([400, 800])  # initial sizes for the widgets

        self.setCentralWidget(self.splitter)

    def on_return_pressed(self):
        text = self.input_line.text()
        print(f"Текст был сохранен: {text}")


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == '__main__':
    main()
