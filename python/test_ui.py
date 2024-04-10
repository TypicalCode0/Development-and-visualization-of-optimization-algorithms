import sys, os

from PyQt6 import uic, QtCore, QtGui
from PyQt6.QtWidgets import *


class VisualisationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('ui/main_2.ui', self)
        self.setWindowTitle("Visualization of optimization algorithms")

        self.constraints = []

        self.comboBox_choose_alg = self.findChild(QComboBox, "comboBox_choose_alg")
        self.textEdit_expression = self.findChild(QTextEdit, "textEdit_expression")
        self.lineEdit_universe_left = self.findChild(QTextEdit, "lineEdit_universe_left")
        self.lineEdit_universe_right = self.findChild(QTextEdit, "lineEdit_universe_right")
        self.lineEdit_new_constraints = self.findChild(QTextEdit, "lineEdit_new_constraints")
        self.listWidget_constraints = self.findChild(QListWidget, "listWidget_constraints")
        self.pushButton_add_constraint = self.findChild(QPushButton, "pushButton_add_constraint")
        self.pushButton_delete_constraint = self.findChild(QPushButton, "pushButton_delete_constraint")
        self.pushButton_clear_constraints = self.findChild(QPushButton, "pushButton_clear_constraints")
        self.pushButton_ready = self.findChild(QPushButton, "pushButton_ready")
        self.pushButton_stop = self.findChild(QPushButton, "pushButton_stop")
        self.pushButton_step = self.findChild(QPushButton, "pushButton_step")
        self.pushButton_steps = self.findChild(QPushButton, "pushButton_steps")
        self.pushButton_exit = self.findChild(QPushButton, "pushButton_exit")

        self.pushButton_stop.hide()

        self.wait()

    def wait(self):
        self.pushButton_exit.clicked.connect(self.close_program)
        self.pushButton_add_constraint.clicked.connect(self.add_constraint)
        self.pushButton_delete_constraint.clicked.connect(self.delete_constraint)
        self.pushButton_clear_constraints.clicked.connect(self.clear_constraints)
        # self.pushButton_ready.clicked.connect(self.get_data)

    def check_errors(self):
        pass

    def add_constraint(self):
        constraint = self.lineEdit_new_constraints.text()
        self.constraints.append(constraint)
        self.lineEdit_new_constraints.clear()
        self.update_list_constraints()

    def delete_constraint(self):
        pass

    def clear_constraints(self):
        self.constraints.clear()
        self.update_list_constraints()

    def update_list_constraints(self):
        pass

    def get_data(self):
        pass

    def close_program(self):
        pass


if __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # app = QtGui.QGuiApplication(sys.argv)
    app = QApplication(sys.argv)
    ex = VisualisationApp()
    ex.show()
    sys.exit(app.exec())
