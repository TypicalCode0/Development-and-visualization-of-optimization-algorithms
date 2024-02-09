#pip install matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import FunctionObj_v1 as func
import numpy as np

class plot:
    def draw(f):
        if len(f.variables) == 1:
            plot.two_dimensional(f)
        elif len(f.variables) == 2:
            plot.three_dimensional(f)

    def two_dimensional(f):
        size_pic, start, end = 100, f.border[0], f.border[1]
        step = (end - start) / size_pic
        x, y = [], []
        i = start
        while i < end:
            rez = f.solve({f.variables[0]:i})
            if (rez is None):
                continue
            else:
                x.append(i)
                y.append(rez)
            i += step
        plt.plot(x, y)
        plt.xlabel(f.variables[0])
        plt.ylabel(f'f{f.variables[0]}')
        plt.show()

    def three_dimensional(f):
        size_pic, start, end = 100, f.border[0], f.border[1]
        step = (end - start) / size_pic
        X = np.arange(start, end, step)
        Y = np.arange(start, end, step)
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        X, Y = np.meshgrid(X, Y)
        Z = [[0] * len(X[0])] * len(X)
        Z = np.array(Z)
        for i in range(len(X)):
            for j in range(len(X[0])):
                z = f.solve({f.variables[0]:X[i][j], f.variables[1]:Y[i][j]})
                if (z is None):
                    Z = np.ma.masked_where((X == X[i][j]) & (Y == Y[i][j]), Z)
                else: 
                    Z[i][j] = z
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        plt.show()
        
    def multidimensional(f):
        pass
