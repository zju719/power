import matplotlib.pyplot as plt
import numpy as np
import os


class Utility(object):
    @staticmethod
    def DTWDistance(s1, s2, w):
        DTW = {}
        w = max(w, abs(len(s1) - len(s2)))
        for i in range(-1, len(s1)):
            for j in range(-1, len(s2)):
                DTW[(i, j)] = float('inf')
        DTW[(-1, -1)] = 0
        for i in range(len(s1)):
            for j in range(max(0, i - w), min(len(s2), i + w)):
                dist = (s1[i] - s2[j]) ** 2
                DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
        return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

    @staticmethod
    def LB_Keogh(s1, s2, r):
        LB_sum = 0
        for ind, i in enumerate(s1):
            lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
            upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

            if i > upper_bound:
                LB_sum = LB_sum + (i - upper_bound) ** 2
            elif i < lower_bound:
                LB_sum = LB_sum + (i - lower_bound) ** 2
        return np.sqrt(LB_sum)

    @staticmethod
    def filter(array, f, r):
        return np.array([f(array[max(idx - r, 0):min(idx + r + 1, len(array))]) for idx, d in enumerate(array)])


class Power(object):
    def __init__(self, data, id):
        self.id = id
        self.data = data
        self.sum = np.sum(data)
        self.ave = np.mean(data)
        self.std = np.std(data)
        self.per = None  # init outside
        self.rank = None  # init outside

    def __getitem__(self, name):
        if name == "is_nodata":
            value = self.ave < 5
        elif name == "is_close_at_end":
            value = all([v < 10 for v in self.data[-15:]])
        elif name == "is_stop_abs":
            value = any([v < 5 for v in Utility.filter(self.data, np.mean, 7)])
        elif name == "is_stop_rel":
            value = any([v < 0.1 * self.ave for v in Utility.filter(self.data, np.mean, 15)])
        elif name == "is_year_similar":
            value = self.similar()
        elif name == "is_big":
            value = self.per > 0.01
        else:
            raise KeyError("No such attr " + name)
        return value

    def plot(self, path):
        if not os.path.exists("./images/" + path):
            os.system("mkdir ./images/" + path)
        plt.figure(figsize=(18, 10))
        plt.plot(self.data[:365])
        plt.plot(self.data[365:])
        name = str(self.id) + ":" + str(self.rank)
        plt.savefig("./images/" + path + "/" + name + ".jpg")
        plt.close()

    def similar(self):
        data_mf = np.array([v for v in Utility.filter(self.data, np.median, 3)])
        r = 3
        s1 = self.norm(data_mf[123:273])
        s2 = self.norm(data_mf[489:639])
        return Utility.DTWDistance(s1, s2, r) < 900

    def norm(self, array):
        s = array - np.mean(array)
        s /= np.std(array)
        return array


class Powers(object):
    """
    Power class
    """

    def __init__(self, file):
        import os
        import cPickle as pickle

        pfile = file.split(".")[0] + ".pkl"
        if os.path.exists(pfile):
            with open(pfile, "rb") as f:
                self.data = pickle.load(f)
        else:
            import pandas as pd
            df = pd.read_csv(file)
            users = [df[df.user_id == user] for user in df.user_id.unique()]
            self.data = np.array([list(user.power_consumption) for user in users]).astype(np.float)
            print self.data.shape
            with open(pfile, "wb") as f:
                pickle.dump(self.data, f)
        self.train = self.data[:, :609]
        self.test = self.data[:, 609:]

        # split train/test
        total = np.sum(self.train)
        self.powers = [Power(d, idx) for idx, d in enumerate(self.train)]
        self.totals = [p.sum for p in self.powers]
        self.rank = self.ranks(self.totals)
        for idx, p in enumerate(self.powers):
            p.per = 1.0 * p.sum / total
            p.rank = self.rank[idx]

    def __getitem__(self, n):
        return self.powers[n]

    def ranks(self, array):
        r = range(len(array))
        array_with_index = zip(array, r)
        sorted_array = sorted(array_with_index, key=lambda x: x[0])
        array_map = zip([s[1] for s in sorted_array], r)
        sorted_index = sorted(array_map, key=lambda x: x[0])
        rank = [s[1] for s in sorted_index]
        return rank

    def classify(self, p):
        for name in ["is_nodata", "is_big", "is_close_at_end", "is_stop_abs", "is_stop_rel", "is_year_similar"]:
            if p[name]:
                p.plot(name)
                return
        p.plot("other")

    def classifyAll(self):
        for p in self.powers:
            self.classify(p)

    def test_case(self, n):
        self.classify(self.powers[n])

p = Powers("Tianchi_power_new.csv")
p.classifyAll()
