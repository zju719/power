import matplotlib.pyplot as plt
import numpy as np
import os
import cPickle as pickle
import arma as ar
import pandas as pd
import csv


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

    @staticmethod
    def plot(array):
        plt.figure()
        plt.plot(array)
        plt.show()


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
            value = value or np.mean(Utility.filter(self.data, np.median, 3)) < 5
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
        plt.plot(self.data[:366])
        plt.plot(self.data[366:])
        name = str(self.id) + ":" + str(self.rank)
        plt.savefig("./images/" + path + "/" + name + ".jpg")
        plt.close()

    def similar(self):
        data_mf = np.array([v for v in Utility.filter(self.data, np.median, 3)])
        r = 3
        s1 = self.norm(data_mf[-486:-366])
        s2 = self.norm(data_mf[-120:])
        similar_value = Utility.DTWDistance(s1, s2, r)
        print(similar_value)
        return similar_value < 10

    def norm(self, array):
        s = array - np.mean(array)
        s /= np.std(array)
        return s


class Powers(object):
    """
    Power class
    """

    def __init__(self, file):
        pfile = file.split("-")[0] + ".pkl"
        if os.path.exists(pfile):
            with open(pfile, "rb") as f:
                self.data = pickle.load(f)
        else:
            df = pd.read_csv(file)
            users = [df[df.user_id == user] for user in df.user_id.unique()]
            self.data = np.array([list(user.power_consumption) for user in users]).astype(np.float)
            print self.data.shape
            with open(pfile, "wb") as f:
                pickle.dump(self.data, f)
        print self.data.shape
        self.train = self.data[:, :]
        self.preprocess()
        # self.test = self.data[:, 609:]
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

    def preprocess(self):
        data = self.train
        # rule1
        data[128][596:614] /= 1.4


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


def group_folder(folder, powers):
    array = [int(f.split(".")[0].split("-")[0]) for f in os.listdir("./images/" + folder)]
    p = np.array([powers[i].data for i in array])
    return np.sum(p, 0)


class Predict(object):
    def __init__(self, npdata, type):
        npadd31 = self.add31days(npdata)
        if type == 1:
            self.predict31 = self.get_predict31_type1(npadd31)
        elif type == 2:
            self.predict31 = self.get_predict31_type2(npadd31)
        elif type == 3:
            self.predict31 = self.get_predict31_type3(npadd31)

    def get_predict31(self):
        return self.predict31

    def add31days(self, npdata):
        return np.concatenate((npdata, np.zeros(31)))

    def get_filter_data(self, npdata):
        data_filter = Utility.filter(npdata, np.median, 10)
        data_filter = Utility.filter(data_filter, np.mean, 7)
        data_filter[-40:] = data_filter[-40 - 366:-366] + data_filter[-40] - data_filter[-40 - 366]
        return data_filter

    def train_arma(self, pddata):
        ar_model = ar.arima_model(pddata[-90 - 31:-31], maxLag=8)
        ar_model.get_proper_model()
        print 'bic:', ar_model.bic, 'p:', ar_model.p, 'q:', ar_model.q
        # print ar_model.properModel.forecast()[0]
        return ar_model

    def get_arma_perdict(self, npdata):
        pddates = pd.date_range('20150101', periods=len(npdata))
        pddata = pd.DataFrame(npdata, index=pddates)
        arma_model = self.train_arma(pddata)
        # predict
        npdata_end = len(npdata)
        ar_delta = arma_model.properModel.predict(str(pddates[npdata_end - 31]), str(pddates[npdata_end - 1]),
                                                  dynamic=True).values
        return ar_delta


    def get_predict31_type1(self, npdata):
        fdata = self.get_filter_data(npdata)
        npdelta = npdata - fdata

        # a = self.get_arma_perdict(npdelta)
        #
        # plt.plot(range(len(npdelta)),npdelta)
        # plt.show()

        return fdata[-31:] + self.get_arma_perdict(npdelta)

    def get_predict31_type2(self, npdata):
        return npdata[-31 - 366 :-366] + npdata[-32] - npdata[-32 - 366]

    def get_predict31_type3(self, npdata):
        return npdata[-31 - 366 :-366]

def write_csv(csv_name, predict_power):
    print('--export date')
    with open(csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['predict_date', 'predict_power_consumption'])
        for date in range(31):
            writer.writerow([20161001 + date, int(predict_power[date])])


if __name__ == '__main__':
    p = Powers("Tianchi_power_new.csv")
    # p.classifyAll()

    # for f in os.listdir("./images"):
    #     print f
    #     npdata = group_folder(f, p)
    #     Utility.plot(npdata)

    # array = []
    # for f in os.listdir("./images"):
    #     dat = group_folder(f, p)
    #     array.append(dat)
    # # np.savetxt("data.csv", np.array(array))
    # np.loadtxt(open("data.csv"))

    predata = np.zeros(31)
    train_total = np.zeros(639)
    pre_total = np.zeros(31)

    type1 = ['is_stop_rel', 'other', 'is_year_similar', 'is_less_similar', 'is_stop_abs', 'is_big_similar']
    type2 = ['is_same_add_bias']
    type3 = ['is_same']

    for f in os.listdir("./images"):
        print(f)
        traindata = group_folder(f, p)
        if f in type1:
            pre = Predict(traindata, 1)
            predata = pre.get_predict31()
        elif f in type2:
            pre = Predict(traindata, 2)
            predata = pre.get_predict31()
        elif f in type3:
            pre = Predict(traindata, 3)
            predata = pre.get_predict31()
        else:
            continue
        train_total += traindata
        pre_total += predata
        # plt.plot(range(len(traindata)),traindata,'k')
        # plt.plot(range(len(traindata),len(traindata)+31), predata, 'b')
        # plt.show()

    pre_total[0] = 2823780
    pre_total[1] = 2768150
    pre_total[2] = 2733020
    pre_total[3] = 2990660

    plt.plot(range(len(train_total)), train_total, 'k')
    plt.plot(range(len(train_total), len(train_total) + 31), pre_total, 'b')
    plt.show()

    write_csv('Tianchi_power_predict_table.csv', pre_total)