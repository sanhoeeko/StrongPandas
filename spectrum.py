import random as rd

from StrongPandasBase import *
from scipy.signal import savgol_filter
from scipy.stats import linregress


def get_rand_rgba(option='normal'):
    rgba = np.random.rand(4, )
    if option == 'normal':
        rgba[3] = 1
    elif option == 'light':
        rgba[3] = 0.5
    else:
        raise TypeError
    return rgba


def gauss(x: np.ndarray, A, mu, sig2):
    return A * np.exp(-(x - mu) ** 2 / (2 * sig2))


def low_percent(arr, percent):
    y = list(arr.copy())
    y.sort()
    return y[int(len(y) * percent)]


def simpleAnneal(agent, param, itertimes):
    """
    params:
    :agent: input param, return energy
    :param: to optimize
    """
    T = 1.0
    alpha = 0.99
    T_min = alpha ** itertimes
    steplength = 2 * 4

    current_state = param
    current_energy = agent(current_state)
    best_state = current_state
    best_energy = current_energy

    while T > T_min:
        for i in range(4):
            new_state = current_state + ((np.random.rand(2, ) - 0.5) * steplength).astype(int)
            new_energy = agent(new_state)
            delta_energy = new_energy - current_energy
            ap = np.exp(-delta_energy / T)
            if delta_energy < 0 or ap > rd.uniform(0, 1):
                current_state = new_state
                current_energy = new_energy
                if current_energy < best_energy:
                    best_state = current_state
                    best_energy = current_energy
        T *= alpha

    return best_state


class Scale:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a * x + self.b

    def std_scale(self, x):
        return self.a * x

    def sqr_scale(self, x):
        return self.a ** 2 * x

    def inv(self, y):
        x = (y - self.b) / self.a
        return round(x)


class Peak:
    def __init__(self, arr, startpoint, scaling=None):
        x = np.arange(0, len(arr), 1)
        a = arr.copy()[~(arr == 0)]
        therihold = low_percent(a, 0.1)
        x = x[~(arr < therihold)]
        y = arr[~(arr < therihold)]
        y = np.log(y)
        z = np.polyfit(x, y, 2)
        self.startpoint = startpoint
        self.length = len(arr)
        self.scaling = scaling
        self.sig2 = -1 / (2 * z[0])
        self.mu = -z[1] / (2 * z[0]) + startpoint
        self.A = np.exp(z[2] - z[1] ** 2 / (4 * z[0]))
        self.half_h_full_w = 2 * np.sqrt(2 * np.log(2) * self.sig2)
        x = np.arange(0, len(arr), 1)
        if self.scaling is not None:
            self.mu = self.scaling(self.mu)
            self.sig2 = self.scaling.sqr_scale(self.sig2)
            self.half_h_full_w = self.scaling.std_scale(self.half_h_full_w)
            x = self.scaling(x)
        y = self(x)
        self.count = np.sum(arr)
        self.mse = np.sum((y - arr) ** 2) / self.length

    def __call__(self, x):
        return gauss(x, self.A, self.mu, self.sig2)

    def __repr__(self):
        return str(obj2dict(self, ['A', 'mu', 'half_h_full_w', 'count', 'mse']))

    def view(self):
        x = np.arange(0, self.length, 1) + self.startpoint
        if self.scaling is not None: x = self.scaling(x)
        y = self(x)
        plt.plot(x, y)

    def score(self):
        return self.mse


class Spectrum(Variable):
    def __init__(self, name):
        if type(name) == Variable:
            self.arr = name.arr
            name = name.name
        super(Spectrum, self).__init__(name)
        self.xaxis = None
        self.scaling = None

    def view(self):
        if self.xaxis is None:
            self.xaxis = Variable('道址')
            self.xaxis.arr = np.arange(0, len(self.arr), 1).astype(int)
        self.xaxis.draw(self, style='dense')

    def showROI(self, *roi: tuple):
        """
        roi is energy(keV) or location
        """
        if self.xaxis is None:
            raise ValueError
        for r in roi:
            a = r[0]
            b = r[1]
            if self.scaling is not None:
                a = self.scaling.inv(a)
                b = self.scaling.inv(b)
            plt.fill_between(self.xaxis.arr[a:b], 0, self.arr[a:b], color=get_rand_rgba('light'))

    def showPtr(self, x):
        """
        x is energy(keV) or location
        """
        self.showROI((x - 1, x + 1))

    def singlePeak(self, a, b):
        return Peak(self.arr[a:b], startpoint=a, scaling=self.scaling)

    def autoPeakfromCenter(self, a, b, a_min=0, b_max=None):
        """
        select a small interval and expand it, minimize mse
        """
        if b_max is None: b_max = len(self.arr)
        aa = a
        bb = b
        right_expand = True
        flag = 0
        max_score = self.singlePeak(a, b).score()
        while True:
            if right_expand:
                s = self.singlePeak(aa, bb + 1).score()
                if s > max_score:
                    max_score = s
                    bb = bb + 1
                    flag = 0
                else:
                    right_expand = False
                    flag += 1
            else:
                s = self.singlePeak(aa - 1, bb).score()
                if s > max_score:
                    max_score = s
                    aa = aa - 1
                    flag = 0
                else:
                    right_expand = True
                    flag += 1
            if flag == 2 or (aa == a_min and bb == b_max):
                break
        return self.singlePeak(aa, bb)

    def autoPeak(self, a, b, itertimes=200):
        """
        use anneal simulation to minimize mse
        """
        aa, bb = simpleAnneal(lambda x: self.singlePeak(*x).score(), np.array([a, b]), itertimes)
        return self.singlePeak(aa, bb)

    def smooth(self, window_length, polyorder):
        """
        packed: scipy.savgol_filter
        """
        self.arr = savgol_filter(self.arr, window_length, polyorder)
        # correct overfit
        self.arr[self.arr < 0] = 0

    def autosmooth(self, eps):
        """
        eps = longer_window_serration / shorter_window_serration, between 0 and 1
        the closer to 0, the smoother.
        """
        y = self.arr.copy()
        serration = []
        ws = np.arange(11, 201, 10)
        for w in ws:
            ysm = savgol_filter(y, w, 3)
            dy_tot = np.sum(np.abs(np.diff(ysm)))
            serration.append(dy_tot)
        serration = np.array(serration)
        serration = -np.diff(np.log(serration))
        idx = None
        for i in range(len(serration)):
            if serration[i] < eps:
                idx = i
                break
        window_length = ws[idx] if idx else 201
        self.smooth(window_length, 3)

    def scale(self, *points):
        if len(points) < 2: raise ValueError
        self.xaxis = Variable('能量(keV)')
        self.xaxis.arr = np.arange(0, len(self.arr), 1)
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        self.scaling = Scale(slope, intercept)
        self.xaxis.arr = self.scaling(self.xaxis.arr)
