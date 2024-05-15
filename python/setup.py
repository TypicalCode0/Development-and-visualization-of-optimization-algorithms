import os
import sys
from PyQt6 import uic
from PyQt6.QtWidgets import *
from sympy import SympifyError
from FunctionObj_v1 import FunctionObj as Func
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Поменяй пути к файлам!

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
        self.pushButton_clear_graphics = self.findChild(QPushButton, "pushButton_clear_graphics")
        self.widget_graphics = self.findChild(QWidget, "widget_graphics")
        self.layout_graphics = self.findChild(QVBoxLayout, "layout_graphics")
        self.checkbox_contour_plot = self.findChild(QCheckBox, "checkbox_contourplot")
        self.multiplier = self.findChild(QLineEdit, "lineEdit_multiplier")
        self.constraints = []
        self.algorithms = ["Gradient Descent", "Sequential Programming", "Interior-point methods"]
        self.curr_alg = 0
        self.coordinates_steps = []

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
        self.pushButton_clear_graphics.clicked.connect(self.clear_layout)
        self.pushButton_step.clicked.connect(self.print_step)
        self.pushButton_steps.clicked.connect(self.print_steps)
        self.index_step = 0
        self.step = 1

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
        self.listWidget_constraints.takeItem(self.listWidget_constraints.currentRow())

    def clear_constraints(self):
        self.constraints.clear()
        self.listWidget_constraints.clear()

    def start_algorithms(self):
        self.pushButton_ready.hide()
        self.pushButton_stop.show()
        self.pushButton_step.setEnabled(True)
        self.pushButton_steps.setEnabled(True)
        self.pushButton_clear_constraints.setEnabled(False)
        self.index_step = 0
        self.get_data()

    def stop_algorithms(self):
        self.pushButton_stop.hide()
        self.pushButton_ready.show()
        self.pushButton_step.setEnabled(False)
        self.pushButton_steps.setEnabled(False)
        self.pushButton_clear_constraints.setEnabled(True)

    def get_data(self):
        expression = self.textEdit_expression.toPlainText()
        print("function is", expression)
        print(f"limitations are: {self.constraints}")
        print(f"universe is: ({self.lineEdit_universe_left.text()};{self.lineEdit_universe_right.text()})")
        try:
            f = Func(expression)
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
        try:
            self.step = float(self.multiplier.text())
        except:
            print("Неправильный формат шага")
            return
        self.draw(f)

    def close_program(self):
        self.clear_layout()
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
        else:
            pass

    def print_step(self):
        if len(self.coordinates_steps) <= self.index_step:
            return
        if len(self.coordinates_steps[self.index_step]) == 2 or self.checkbox_contour_plot.isChecked():
            self.ax.scatter(float(self.coordinates_steps[self.index_step][0]),
                            float(self.coordinates_steps[self.index_step][1]), marker="^")
        elif len(self.coordinates_steps[self.index_step]) == 3:
            self.ax.scatter(float(self.coordinates_steps[self.index_step][0]),
                            float(self.coordinates_steps[self.index_step][1]),
                            self.coordinates_steps[self.index_step][2], c="black", marker="*", alpha=1, s=50)
        self.canvas.draw()
        self.canvas.flush_events()
        self.index_step += 1

    def print_steps(self):
        for i in range(self.index_step, len(self.coordinates_steps)):
            if len(self.coordinates_steps[i]) == 2 or self.checkbox_contour_plot.isChecked():
                self.ax.scatter(float(self.coordinates_steps[i][0]), float(self.coordinates_steps[i][1]), marker="^")
            elif len(self.coordinates_steps[i]) == 3:
                self.ax.scatter(float(self.coordinates_steps[i][0]), float(self.coordinates_steps[i][1]),
                                self.coordinates_steps[i][2], c="black", marker="*", alpha=1, s=50)
            self.canvas.draw()
            self.canvas.flush_events()

    def calculate_steps_algorithm(self, f):
        file = open("tmp.txt", "r")
        coordinate = file.readline()
        self.coordinates_steps.clear()
        while coordinate:
            if coordinate != "":
                coordinate = coordinate.split()
                if len(coordinate) == 1:
                    coordinate.append(f.solve({f.variables[0]: float(coordinate[0])}))
                elif len(coordinate) == 2:
                    coordinate.append(
                        f.solve({f.variables[0]: float(coordinate[0]), f.variables[1]: float(coordinate[1])}))
                self.coordinates_steps.append(coordinate)
            coordinate = file.readline()
        file.close()
        os.remove("tmp.txt")

    def two_dimensional(self, f):
        size_pic, start, end = 100, f.border[0], f.border[1]
        step = (end - start) / size_pic
        x, y = [], []
        i = start
        while i < end:
            rez = f.solve({f.variables[0]: i})
            i += step
            if rez is None:
                continue
            else:
                x.append(i)
                y.append(rez)
        self.fig = Figure()
        self.ax = self.fig.add_subplot()
        self.ax.plot(x, y)
        self.ax.set_xlabel(f.variables[0])
        self.ax.set_ylabel(f"f({f.variables[0]})")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.graph_layout.addWidget(self.canvas)
        subprocess.run(["../cpp/build/gd.exe",
                        f"{f.exp}", f"{start}", f"{end}", f"{self.step}", "60"], check=True)
        self.calculate_steps_algorithm(f)

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
                z = f.solve({f.variables[0]: X[i][j], f.variables[1]: Y[i][j]})
                if z is None:
                    Z = np.ma.masked_where((X == X[i][j]) & (Y == Y[i][j]), Z)
                else:
                    Z[i][j] = z
        if self.checkbox_contour_plot.isChecked():
            self.fig, self.ax = plt.subplots()
            self.ax.contour(X, Y, Z)
        else:
            self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
            self.ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.8)
            self.ax.set_zlabel(f"f({f.variables[0]},{f.variables[1]})")
        self.ax.set_xlabel(f.variables[0])
        self.ax.set_ylabel(f.variables[1])
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.graph_layout.addWidget(self.canvas)
        subprocess.run(["../cpp/build/gd.exe",
                        f"{f.exp}", f"{start}", f"{end}", f"{self.step}", "60"], check=True)
        self.calculate_steps_algorithm(f)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = VisualisationApp()
    ex.show()
    sys.exit(app.exec())
