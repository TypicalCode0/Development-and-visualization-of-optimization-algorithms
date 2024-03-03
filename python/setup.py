#pip install pyqt6

from plotting import plot
from FunctionObj_v1 import FunctionObj as func
from PyQt6.QtWidgets import QApplication, QMainWindow, QSplitter, QWidget, QLineEdit, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QSplitter, QWidget, QLineEdit, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6 import QtCore, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Настройка окна
        self.setGeometry(100, 100, 1200, 800)  # x, y, width, height
        self.setWindowTitle('course_work')

        # Создание сплиттера и виджетов
        self.splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: black }")

        self.widget1 = QtWidgets.QWidget(self.splitter)
        self.widget1.setStyleSheet("background-color: white; border: 2px solid black;")

        self.widget2 = QtWidgets.QWidget(self.splitter)
        self.widget2.setStyleSheet("background-color: grey; border: 2px solid black;")

        # Размеры виджетов
        self.splitter.setSizes([400, 800])  # initial sizes for the widgets

        # Центральный виджет
        self.setCentralWidget(self.splitter)

        # Вертикальный layout для первого виджета
        self.layout1 = QtWidgets.QVBoxLayout(self.widget1)
        self.layout1.setContentsMargins(10, 10, 10, 10)
        self.layout1.setSpacing(0)



        # Текст поля функции
        self.label = QtWidgets.QLabel("Enter the function:", self.widget1)
        self.label.setStyleSheet("border: none;")
        self.layout1.addWidget(self.label)

        # Поле функции
        self.input_line = QtWidgets.QLineEdit(self.widget1)
        self.input_line.setStyleSheet("border: 1px solid black;")
        self.input_line.returnPressed.connect(self.on_return_pressed)
        self.layout1.addWidget(self.input_line)


        # Горизонтальные layout'ы для текстов и полей limitations и universe
        h1_layout = QtWidgets.QHBoxLayout()
        h2_layout = QtWidgets.QHBoxLayout()

        # Текст поля Limitations
        self.label_limitations = QtWidgets.QLabel("Limitations:", self.widget1)
        self.label_limitations.setStyleSheet("border: none;")

        h1_layout.addWidget(self.label_limitations)

        # Поле Limitations
        self.input_limitations = QtWidgets.QLineEdit(self.widget1)
        self.input_limitations.setStyleSheet("border: 1px solid black;")
        self.input_limitations.returnPressed.connect(self.on_return_pressed)
        h2_layout.addWidget(self.input_limitations)

        # Текст поля Universe
        self.label_universe = QLabel("Universe:", self.widget1)
        self.label_universe.setStyleSheet("border: none;")
        h1_layout.addWidget(self.label_universe)

        # Поле Universe
        self.input_universe = QLineEdit(self.widget1)
        self.input_universe.setStyleSheet("border: 1px solid black;")
        self.input_universe.returnPressed.connect(self.on_return_pressed)
        h2_layout.addWidget(self.input_universe)



        # Добавляем горизонтальный layout в общий вертикальный layout
        self.layout1.addLayout(h1_layout)
        self.layout1.addLayout(h2_layout)

        # Растягиваем пространство
        self.layout1.addStretch()

    def on_return_pressed(self):
        text = self.input_line.text()
        print(f"Текст был сохранен: {text}")
        f = func(text)
        left_border, right_border = list(map(int,self.input_universe.text().split(';')))
        f.add_border(left_border, right_border)
        limit = self.input_limitations.text()
        f.add_constraint(limit)
        plot.draw(f)
        f = func(text)



def main():

    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()

if __name__ == '__main__':
    main()
