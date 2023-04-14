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
        self.arr = None  # -*- coding: utf-8 -*-


# Chinese display in matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# all variables are cited in the pool, then you can update them in one command
_pool = []


class Variable:
    def __init__(self, name):
        self.name = name
        self.arr = None
        self.func = None
        self.depend = None
        self.require = []
        _pool.append(self)

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
        if type(x) != Variable:
            x = constant(x)
        return dependent_variable(lambda x, y: x + y, [self, x])

    def __radd__(self, x):
        return self + x

    def __sub__(self, x):
        if type(x) != Variable:
            x = constant(x)
        return dependent_variable(lambda x, y: x - y, [self, x])

    def __rsub__(self, x):
        return self * (-1) + x

    def __mul__(self, x):
        if type(x) != Variable:
            x = constant(x)
        return dependent_variable(lambda x, y: x * y, [self, x])

    def __rmul__(self, x):
        return self * x

    def __truediv__(self, x):
        if type(x) != Variable:
            x = constant(x)
        return dependent_variable(lambda x, y: x / y, [self, x])

    def __rtruediv__(self, x):
        return self.to(lambda x: 1 / x) * x

    def __pow__(self, x):
        if type(x) != Variable:
            x = constant(x)
        return dependent_variable(lambda x, y: x ** y, [self, x])

    def __repr__(self):
        return str(self.arr)

    def __getitem__(self, i):
        # auto evaluate
        if self.arr is None:
            self.eval()
        return self.arr[i]

    def draw(self, *ys: 'Variable'):
        # draw as x_variable
        for y in ys:
            plt.plot(self.arr, y.arr, '.-')
        plt.legend(list(map(lambda x: x.name, ys)))
        plt.xlabel(self.name)

    def to(self, func):
        # compatible with other transformation, like sin, exp, etc.
        res = Variable(None)
        res.depend = [self]
        res.func = func
        return res


def get_var_from_csv(filename, list_of_vars, encoding='utf-8'):
    df = pd.read_csv(filename, encoding=encoding)
    for v in list_of_vars:
        v.get_data(df)


def eval_all():
    for v in _pool:
        if not v.require:
            v.eval()


def dependent_variable(func, depend: list[Variable]):
    v = Variable(None)
    v.func = func
    v.depend = depend
    for d in depend:
        d.require.append(v)
    return v


def constant(const):
    v = Variable(None)
    v.arr = const
    return v


def pile(varlst: list[Variable]):
    v = Variable(None)
    v.arr = np.hstack(list(map(lambda x: x.arr, varlst)))
    return v


def get_significant_digits(val: float, digit: int):
    return np.format_float_positional(val, precision=digit, unique=False, fractional=False, trim='k')


def vars_to_latex_table(varlst: list[Variable], digits=None):
    r"""
    Template:
    % \begin{table}[H]
    %   \centering
    %     \caption{}\label{}
    %     \begin{tabular}{c*N}
    %         \toprule[1.5pt]
    %         \\
    %         \midrule[.5pt]
    %         \\
    %         \bottomrule[1.5pt]
    %     \end{tabular}
    % \end{table}
    """
    if digits is None:
        digits = [4] * len(varlst)
    for v in varlst:
        v.eval()
    names = list(map(lambda x: x.name, varlst))
    arrs = list(map(lambda x: x.arr, varlst))
    cs = len(varlst)
    head = r'''
    \begin{table}[H]
    \centering
    \caption{}\label{}
    \begin{tabular}{''' + 'c' * cs + r'''}
    \toprule[1.5pt]
    '''
    tail = r'''
    \bottomrule[1.5pt]
    \end{tabular}
    \end{table}
    '''
    mid = r'\midrule[.5pt]'
    names = '& '.join(names) + '\\\\\n'
    arrs = np.array(arrs).T
    body = ''
    for i in range(len(arrs)):
        for j in range(cs):
            body += str(get_significant_digits(arrs[i, j], digits[j])) + '& '
        body = body[:-2] + '\\\\\n'
    return '\n'.join([head, names, mid, body, tail])

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
    def draw(self, *ys: 'Variable'):
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
