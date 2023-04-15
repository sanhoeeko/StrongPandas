# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chinese display in matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# all variables are cited in the pool, then you can update them in one command
_pool = []


class Variable:
    def __init__(self, name):
        self.name = name
        self.arr = None
        self.other_info = []
        self.func = None
        self.depend = None
        self.require = []
        _pool.append(self)

    def get_data(self, df):

        # if the csv doesn't contain name, set the name an empty string,
        # then StrongPandas will automaticly find the first column
        if self.name == '':
            series = df.iloc[:, 0]
        elif isinstance(self.name, int):
            series = df.iloc[:, self.name]
        else:
            series = df[self.name]

        if series.isnull().values.any():
            raise Exception('There is nan in variable: ' + str(self.name))

        if isinstance(series[0], (int, float, complex)):
            self.arr = np.array(series)
        else:
            # if there are some lines that contains non-numerical messages
            for i in range(len(series)):
                if series[i].isdigit():
                    self.arr = np.array(series[i:], dtype=float)
                    if i > 0:
                        self.other_info.append(list(series[:i]))
                    return

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

    def make_info_dict(self, separator=': '):
        if self.other_info is not None:
            for i in range(len(self.other_info)):
                lst = list(map(lambda x: x.split(
                    separator), self.other_info[i]))
                self.other_info[i] = dict(lst)

    def get_info(self, key):
        def auto_todigit(val):
            if type(val) == str:
                if val.isdigit():
                    return float(val)
            return val

        if type(self.other_info[0]) != dict:
            self.make_info_dict()
        if len(self.other_info) == 1:
            return auto_todigit(self.other_info[0][key])
        else:
            res = []
            for d in self.other_info:
                res.append(auto_todigit(d[key]))
            return res

    def plot(self):
        plt.plot(self.arr, '.-')

    def hist(self, bins=20):
        plt.hist(self.arr, bins=bins)

    def to_csv(self, filename):
        arr = pd.Series(self.arr)
        head = pd.Series(self.other_info)
        head = head.append(arr)
        head.to_csv(filename, header=None, index=False)


def get_var_from_csv(filename, list_of_vars, ifHeader='auto', encoding='utf-8'):
    # if the name of variable is not the name of column, it may regard the first line of csv as data
    if ifHeader == 'auto':
        ifHeader = True
        for v in list_of_vars:
            n = v.name
            if isinstance(n, int) or n == '':
                ifHeader = False
                break
    if ifHeader:
        df = pd.read_csv(filename, encoding=encoding)
    else:
        df = pd.read_csv(filename, encoding=encoding, header=None)
    for v in list_of_vars:
        v.get_data(df)


def eval_all():
    for v in _pool:
        if not v.require:
            v.eval()


def dict_all():
    for v in _pool:
        v.make_info_dict()


def dependent_variable(func, depend: list[Variable]):
    v = Variable(None)
    v.func = func
    v.depend = depend
    for d in depend:
        d.require.append(v)
        v.other_info.extend(d.other_info)
    return v


def constant(const):
    v = Variable(None)
    v.arr = const
    return v


def pile(varlst: list[Variable]):
    v = Variable(None)
    v.arr = np.hstack(list(map(lambda x: x.arr, varlst)))
    return v


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

    def get_significant_digits(val: float, digit: int):
        return np.format_float_positional(val, precision=digit, unique=False, fractional=False, trim='k')

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
