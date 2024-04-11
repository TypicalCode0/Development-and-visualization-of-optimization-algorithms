import sys
import time
from PyQt6 import uic, QtCore, QtGui
from PyQt6.QtWidgets import *
from sympy import SympifyError
from FunctionObj_v1 import FunctionObj as func
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class VisualisationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/main.ui", self)
        self.setWindowTitle("Visualization of optimization algorithms")

        self.comboBox_choose_alg = self.findChild(QComboBox, "comboBox_choose_alg")
        self.textEdit_expression = self.findChild(QTextEdit, "textEdit_expression")
        self.lineEdit_universe_left = self.findChild(QLineEdit, "lineEdit_universe_left")
        self.lineEdit_universe_right = self.findChild(QLineEdit, "lineEdit_universe_right")
        self.lineEdit_new_constraints = self.findChild(QLineEdit, "lineEdit_new_constraints")
        self.listWidget_constraints = self.findChild(QListWidget, "listWidget_constraints")
        self.pushButton_add_constraint = self.findChild(QPushButton, "pushButton_add_constraint")
        self.pushButton_delete_constraint = self.findChild(QPushButton, "pushButton_delete_constraint")
        self.pushButton_clear_constraints = self.findChild(QPushButton, "pushButton_clear_constraints")
        self.pushButton_ready = self.findChild(QPushButton, "pushButton_ready")
        self.pushButton_step = self.findChild(QPushButton, "pushButton_step")
        self.pushButton_steps = self.findChild(QPushButton, "pushButton_steps")
        self.pushButton_exit = self.findChild(QPushButton, "pushButton_exit")
        self.pushButton_stop = self.findChild(QPushButton, "pushButton_stop")
        self.widget_graphics = self.findChild(QWidget, "widget_graphics")
        self.layout_graphics = self.findChild(QVBoxLayout, "layout_graphics")

        self.constraints = []
        self.algorithms = ["Gradient Descent", "1"]
        self.curr_alg = 0

        self.graph_layout = QVBoxLayout(self.widget_graphics)
        for i in self.algorithms:
            self.comboBox_choose_alg.addItem(i)
        self.comboBox_choose_alg.currentIndexChanged.connect(self.change_status_constraints_widgets)

        self.pushButton_stop.hide()
        self.pushButton_exit.clicked.connect(self.close_program)
        self.pushButton_add_constraint.clicked.connect(self.add_constraint)
        self.pushButton_delete_constraint.clicked.connect(self.delete_constraint)
        self.pushButton_clear_constraints.clicked.connect(self.clear_constraints)
        self.pushButton_ready.clicked.connect(self.start_algorithms)
        self.pushButton_stop.clicked.connect(self.stop_algorithms)

    def change_status_constraints_widgets(self):
        current = int(self.comboBox_choose_alg.currentIndex())
        if current == 0:
            self.listWidget_constraints.setEnabled(False)
            self.pushButton_add_constraint.setEnabled(False)
            self.pushButton_delete_constraint.setEnabled(False)
            self.lineEdit_new_constraints.setEnabled(False)
        else:
            self.listWidget_constraints.setEnabled(True)
            self.pushButton_add_constraint.setEnabled(True)
            self.pushButton_delete_constraint.setEnabled(True)
            self.lineEdit_new_constraints.setEnabled(True)

    def check_errors(self):
        pass

    def add_constraint(self):
        constraint = self.lineEdit_new_constraints.text()
        self.constraints.append(constraint)
        self.listWidget_constraints.addItem(constraint)
        self.lineEdit_new_constraints.clear()

    def delete_constraint(self):
        curr_item = self.listWidget_constraints.takeItem(self.listWidget_constraints.currentRow())
        del curr_item

    def clear_constraints(self):
        self.constraints.clear()
        self.listWidget_constraints.clear()

    def start_algorithms(self):
        self.pushButton_ready.hide()
        self.pushButton_stop.show()
        self.pushButton_step.setEnabled(True)
        self.pushButton_steps.setEnabled(True)
        self.get_data()

    def stop_algorithms(self):
        self.pushButton_stop.hide()
        self.pushButton_ready.show()
        self.pushButton_step.setEnabled(False)
        self.pushButton_steps.setEnabled(False)

    def get_data(self):
        expression = self.textEdit_expression.toPlainText()
        print("function is", expression)
        print(f"limitations are: {self.constraints}")
        print(f"universe is: ({self.lineEdit_universe_left.text()};{self.lineEdit_universe_right.text()})")
        try:
            f = func(expression)
        except SympifyError:
            print("Sympify Error")
            return
        try:
            f.add_border(int(self.lineEdit_universe_left.text()), int(self.lineEdit_universe_right.text()))
        except ValueError:
            print("ValueError")
            return
        for limit in self.constraints:
            f.add_constraint(limit)
        self.draw(f)

    def close_program(self):
        sys.exit()

    def clear_layout(self):
        while self.graph_layout.count():
            item = self.graph_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def draw(self, f):
        if len(f.variables) == 1:
            self.two_dimensional(f)
        elif len(f.variables) == 2:
            self.three_dimensional(f)

    def print_steps_algorithm(self, ax, f, fig):
        file = open("tmp.txt", 'r')
        coordinate = file.readline()
        while coordinate:
            if coordinate != "":
                coordinate = coordinate.split()
                if len(coordinate) == 1:
                    ax.scatter(float(coordinate[0]), f.solve({f.variables[0]:float(coordinate[0])}), marker='^')
                elif len(coordinate) == 2:
                    ax.scatter(float(coordinate[0]), float(coordinate[1]), f.solve({f.variables[0]:float(coordinate[0]), f.variables[1]:float(coordinate[1])}), c="#000000", marker='^')
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(1)
            coordinate = file.readline()

    def two_dimensional(self, f):
        size_pic, start, end = 100, f.border[0], f.border[1]
        step = (end - start) / size_pic
        x, y = [], []
        i = start
        while i < end:
            rez = f.solve({f.variables[0]:i})
            i += step
            if (rez is None):
                continue
            else:
                x.append(i)
                y.append(rez)
        fig = Figure()
        ax = fig.add_subplot()
        ax.plot(x, y)
        ax.set_xlabel(f.variables[0])
        ax.set_ylabel(f'f({f.variables[0]})')
        canvas = FigureCanvasQTAgg(fig)
        self.graph_layout.addWidget(canvas)
        subprocess.run(['cpp/bin/gd.exe',
                        f"{f.exp}", f'{start}', f'{end}', f'{step}', "10"], check=True)
        self.print_steps_algorithm(ax, f, fig) 

    def three_dimensional(self, f):
        size_pic, start, end = 100, f.border[0], f.border[1]
        step = (end - start) / size_pic
        X = np.arange(start, end, step)
        Y = np.arange(start, end, step)
        X, Y = np.meshgrid(X, Y)
        Z = [[0] * len(X[0])] * len(X)
        Z = np.array(Z).astype(np.float64)
        for i in range(len(X)):
            for j in range(len(X[0])):
                z = f.solve({f.variables[0]:X[i][j], f.variables[1]:Y[i][j]})
                if (z is None):
                    Z = np.ma.masked_where((X == X[i][j]) & (Y == Y[i][j]), Z)
                else: 
                    Z[i][j] = z
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.set_xlabel(f.variables[0])
        ax.set_ylabel(f.variables[1])
        ax.set_zlabel(f'f({f.variables[0]},{f.variables[1]})')

        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        sc = FigureCanvasQTAgg(fig)
        self.graph_layout.addWidget(sc)
        subprocess.run(['cpp/bin/gd.exe',
                        f"{f.exp}", f'{start}', f'{end}', f'{step}', "10"], check=True)
        self.print_steps_algorithm(ax, f, fig)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VisualisationApp()
    ex.show()
    sys.exit(app.exec())
