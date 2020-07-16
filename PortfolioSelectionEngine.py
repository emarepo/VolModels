import pandas as pd
import numpy as np
import datetime as dt
from math import sqrt
import matplotlib.pyplot as plt

###

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


class PFSE:
    """

    """

    def __init__(self, assets_hist, weights, ret_column):
        """

        :param assets:
        :param weights:

        """
        self.assets = assets_hist
        self.weights = weights
        self.ret_col = ret_column

    def offset_date(self, t, asset_n, return_counter=False):
        """

        :param t: How many days we want to span from today (<0 means we are going back)
        :param return_counter: In some cases it will be useful to know how many days the function
                               will span to get a valid date.
                               When this parameter is set to True, the function will return exactly this value,
                               permitting us to define differences between business days more easily.
                               For example, if we input a Sunday and set return_counter=True, the function will
                               return 2, i.e. the span between Sunday date and Friday date
                               (if Friday is not a holiday).

        :return: business day (if return_counter=False)
        :return: difference between date in t and last valid business date (if return_counter=True)

        """

        d = dt.date.today() + dt.timedelta(days=t)

        counter = 0
        while True:
            if d.weekday() < 5 and d in self.assets[asset_n].index:
                break
            else:
                d = d - dt.timedelta(days=1)
                if return_counter is True:
                    counter += 1
                    continue

        if return_counter is True:
            return counter
        else:
            return d

    def pf_single_return(self, t, asset_n):
        """

        :param t:
        :return:

        """
        ret_list = []
        for i in self.assets:
            ret_list.append(i.loc[self.offset_date(t, asset_n), self.ret_col])
        try:
            ret = np.dot(np.array(ret_list), np.array(self.weights))
        except TypeError:
            ret_fixed = [float(i) for i in ret_list]
            ret = np.dot(np.array(ret_fixed), np.array(self.weights))

        return ret

    def pf_single_volatility(self, t, asset_n):
        """

        :param t:
        :param asset_n:
        :return:

        """

        ret_list = []
        for i in self.assets:
            ret_list.append(float(i.loc[self.offset_date(t, asset_n), self.ret_col]) ** 2)

        w_prod = [2 * w_i * w_j for w_i in self.weights for w_j in self.weights
                  if self.weights.index(w_i) != self.weights.index(w_j)]

        ret_prod = [r_i * r_j for r_i in self.weights for r_j in self.weights
                    if self.weights.index(r_i) != self.weights.index(r_j)]

        var = np.dot(np.array(ret_list), np.array(self.weights)) + np.dot(np.array(ret_prod), np.array(w_prod))

        return sqrt(var)

    def pf_returns(self):
        """

        :return:

        """
        ret = {"Date": [], "Log Returns": []}
        for i in range(100, 3600):
            ret["Date"].append(self.offset_date(-i, 0))
            ret["Log Returns"].append(self.pf_single_return(-i, 0))

        df = pd.DataFrame(ret)
        df.set_index("Date", inplace=True)

        return df

    def pf_volatility(self):
        """

        :return:

        """

        vol = {"Date": [], "Volatility": []}
        for i in range(100, 3600):
            vol["Date"].append(self.offset_date(-i, 0))
            vol["Volatility"].append(self.pf_single_volatility(-i, 0))

        df = pd.DataFrame(vol)
        df.set_index("Date", inplace=True)

        return df

# if __name__ == "__main__":
#     ads = pd.read_csv(r"C:\Users\alfa8\Python Projects\Master venv\Data\Equities\ADS.DE.csv",
#                       parse_dates = ["Date"])[["Date", "Close", "Log Returns"]]
#     ads.dropna(inplace=True)
#     ads.set_index("Date", inplace=True)
#
#     alv = pd.read_csv(r"C:\Users\alfa8\Python Projects\venv\Data\Equities\ALV.DE.csv",
#                       parse_dates=["Date"])[["Date", "Close", "Log Returns"]]
#     alv.dropna(inplace=True)
#     alv.set_index("Date", inplace=True)
#
#     bas = pd.read_csv(r"C:\Users\alfa8\Python Projects\venv\Data\Equities\BAS.DE.csv",
#                       parse_dates=["Date"])[["Date", "Close", "Log Returns"]]
#     bas.dropna(inplace=True)
#     bas.set_index("Date", inplace=True)
#
#     bayn = pd.read_csv(r"C:\Users\alfa8\Python Projects\venv\Data\Equities\BAYN.DE.csv",
#                       parse_dates=["Date"])[["Date", "Close", "Log Returns"]]
#     bayn.dropna(inplace=True)
#     bayn.set_index("Date", inplace=True)
#
#     bmw = pd.read_csv(r"C:\Users\alfa8\Python Projects\venv\Data\Equities\BMW.DE.csv",
#                       parse_dates=["Date"])[["Date", "Close", "Log Returns"]]
#     bmw.dropna(inplace=True)
#     bmw.set_index("Date", inplace=True)
#
#     dai = pd.read_csv(r"C:\Users\alfa8\Python Projects\venv\Data\Equities\DAI.DE.csv",
#                       parse_dates=["Date"])[["Date", "Close", "Log Returns"]]
#     dai.dropna(inplace=True)
#     dai.set_index("Date", inplace=True)
#
#     dte = pd.read_csv(r"C:\Users\alfa8\Python Projects\venv\Data\Equities\DTE.DE.csv",
#                       parse_dates=["Date"])[["Date", "Close", "Log Returns"]]
#     dte.dropna(inplace=True)
#     dte.set_index("Date", inplace=True)
#
#     sap = pd.read_csv(r"C:\Users\alfa8\Python Projects\venv\Data\Equities\SAP.DE.csv",
#                       parse_dates=["Date"])[["Date", "Close", "Log Returns"]]
#     sap.dropna(inplace=True)
#     sap.set_index("Date", inplace=True)
#
#     sie = pd.read_csv(r"C:\Users\alfa8\Python Projects\venv\Data\Equities\SIE.DE.csv",
#                       parse_dates=["Date"])[["Date", "Close", "Log Returns"]]
#     sie.dropna(inplace=True)
#     sie.set_index("Date", inplace=True)
#
#     vow3 = pd.read_csv(r"C:\Users\alfa8\Python Projects\venv\Data\Equities\VOW3.DE.csv",
#                       parse_dates=["Date"])[["Date", "Close", "Log Returns"]]
#     vow3.dropna(inplace=True)
#     vow3.set_index("Date", inplace=True)
#
#     TestBasket = PFSE([ads, alv, bas, bayn, bmw, dai, dte, sap, sie, vow3],
#                       [1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10],
#                         "Log Returns")
#
#     print(TestBasket.pf_single_return(-50,0))
#     print(TestBasket.pf_single_volatility(-50,0))
#
#     plt.figure(1)
#     plt.subplot(211)
#     plt.plot(TestBasket.pf_returns())
#     plt.subplot(212)
#     plt.plot(TestBasket.pf_volatility())
#
#     plt.show()
