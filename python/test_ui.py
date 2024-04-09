import sys

from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtWidgets import *


class MyWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('ui/main.ui', self)
        self.setWindowTitle("AimTrainer")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    ex.show()
    sys.exit(app.exec_())
