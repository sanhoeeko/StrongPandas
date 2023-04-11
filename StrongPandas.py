# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chinese display in matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Variable:
    def __init__(self, name):
        self.name = name
        self.arr = None
        self.func = None
        self.depend = None

    def get_data(self, df):
        series = df[self.name]
        if series.isnull().values.any():
            raise Exception('There is nan in variable: ' + self.name)
        self.arr = np.array(series)

    def eval(self):
        if self.depend is None:
            return
        for d in self.depend:
            d.eval()
        self.arr = self.func(*list(map(lambda x: x.arr, self.depend)))

    def asname(self, name):
        self.name = name

    # operator is for constructing calculation graph
    def __add__(self, x):
        return dependent_variable(lambda x, y: x + y, [self, x])

    def __sub__(self, x):
        return dependent_variable(lambda x, y: x - y, [self, x])

    def __mul__(self, x):
        return dependent_variable(lambda x, y: x * y, [self, x])

    def __truediv__(self, x):
        return dependent_variable(lambda x, y: x / y, [self, x])

    # draw as x_variable
    def draw(self, *ys:'Variable'):
        for y in ys:
            plt.plot(self.arr, y.arr, '.-')
        plt.legend(list(map(lambda x: x.name, ys)))
        plt.xlabel(self.name)


def get_var_from_csv(filename, list_of_vars, encoding='utf-8'):
    df = pd.read_csv(filename, encoding=encoding)
    for v in list_of_vars:
        v.get_data(df)


def dependent_variable(func, depend):
    v = Variable(None)
    v.func = func
    v.depend = depend
    return v


def constant(const):
    v = Variable(None)
    v.arr = const
    return v
