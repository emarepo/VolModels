import pandas as pd
import numpy as np
import datetime as dt
from math import sqrt
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from arch import arch_model
from scipy.linalg import cholesky as chol

###

register_matplotlib_converters()
np.random.seed(1995)

"""

===================================================================================================================

                                                         Engine

===================================================================================================================

"""


class VolTaEngine:
    """

    """

    def __init__(self,
                 ul_history,
                 exp_cap,
                 vol_target,
                 risk_free,
                 close_prices_col="Close",
                 close_ret_col="Returns",
                 intensity=1):
        """

        :param ul_history:
        :param exp_cap:
        :param vol_target:
        :param risk_free:
        :param close_prices_col:
        :param close_ret_col:

        """
        self.ul_hist = ul_history
        self.close_prices_col = close_prices_col
        self.close_ret_col = close_ret_col
        self.exp_cap = exp_cap
        self.vt = vol_target
        self.risk_free = risk_free
        self.intensity = intensity

    def offset_date(self, t, return_counter=False):
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
            if d.weekday() < 5 and d in self.ul_hist.index:
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

    def get_excess_return(self, t):
        """

        :param t:
        :return:

        """
        pass

    def ul_price(self, t):
        """

        :param t:
        :return:

        """
        ul_price = self.ul_hist.loc[self.offset_date(t), self.close_prices_col]

        return ul_price

    def get_ul_return(self, t):
        """

        :param t:
        :return:

        """
        ret = self.ul_price(t) / self.ul_price(t - 1) - 1

        return ret

    """
    
    =====================================================================
    
        Realized Volatility and Exposure Calculation methods:
    
    =====================================================================
        
    """

    def single_theoretical_real_vol(self, t, n):
        """

        :param t:
        :param n:
        :return:

        """
        var = 0
        for i in range(0, n):
            var += self.get_ul_return(t - i) ** 2

        vol = sqrt(var)

        return vol

    def single_theoretical_exposure(self, t, n):
        """

        :param t:
        :param n:
        :return:

        """
        exp = min(self.exp_cap, self.vt / self.single_theoretical_real_vol(t, n))

        return exp

    def single_theoretical_strat_return(self, t, n):
        """

        :param t:
        :param n:
        :return:

        """
        ret = self.single_theoretical_exposure(t, n) * self.get_ul_return(t) + \
              (1 - self.single_theoretical_exposure(t, n)) * self.risk_free

        return ret

    def simple_real_var(self):
        """

        :return:

        """
        ret = []
        for v in self.ul_hist[self.close_ret_col]:
            ret.append((v ** 2 * 252))

        return ret

        pass

    def simple_real_vol(self, returns):
        """

        :return:

        """
        # self.ul_hist[self.close_ret_col]
        vol = []
        for v in returns:
            vol.append(sqrt(v ** 2 * 252))
        #
        # print(vol)
        # plt.plot(vol, color="red")
        # plt.show()
        #
        return vol

    def lin_weight_real_vol(self, returns, gamma, alpha, n):
        """

        :return:

        """
        ret = returns
        alpha_weights = [0.88 * alpha, (alpha - 0.88 * alpha) / 2,
                         (alpha - 0.88 * alpha - ((alpha - 0.88 * alpha) / 2)) / 2] + \
                        (n - 3) * [(alpha - 0.88 * alpha - (alpha - 0.88 * alpha) / 2
                                    - (alpha - 0.88 * alpha - ((alpha - 0.88 * alpha) / 2)) / 2) / 27]
        # print(alpha_weights)
        varlong = []
        var = []

        for v in range(0, len(ret)):
            varlong.append(1 / (v + 1) * sum([i ** 2 for i in ret[:(v + 1)]]))
        for v in range(0, len(ret)):
            if v > n - 1:
                sum_product = sum([x * y for x, y in zip(alpha_weights, [i ** 2 for i in ret[(v - (n - 1)):v]])])
                # print(ret[(v-29):v]**2)
                var.append(sum_product)
            else:
                var.append(ret[v] ** 2)

        weight_vlong = [gamma * i for i in varlong]
        vol = [sqrt((x + y) * 252) for x, y in zip(weight_vlong, var)]

        # print(vol)
        # plt.plot(vol, color="red")
        # plt.show()
        #
        return vol

    def ewma_real_vol(self, returns, lmbda):
        """

            :return:

        """
        ret = returns
        ewma_squared = []

        for v in range(0, len(ret)):
            if v > 0:
                var = lmbda * ewma_squared[:][v - 1] + (1 - lmbda) * ret[v] ** 2
                ewma_squared.append(var)
            else:
                ewma_squared.append(ret[v] ** 2)

        ewma = [sqrt(i * 252) for i in ewma_squared]
        #
        # print(ewma)
        # plt.plot(ewma, color="red")
        # plt.show()

        return ewma

    def exposure(self, realized_volatility):
        """

        :return:

        """

        exp = []
        for vol in realized_volatility:
            exp.append(min(self.exp_cap, self.vt / vol))

        return exp

    """
    
    =====================================================================

                         Returns Shocks Modeling:
    
    =====================================================================
    
    """

    def norm_shock(self):
        """

        :param intensity: Multiplier for the shock intensity, should be changed carefully.
        :return: An array of returns with higher fluctuations than the original ones, simulating outlier events.

        """

        # Calculating rolling standard deviation from the data
        sigma = self.ul_hist[self.close_ret_col].rolling(50).std()

        # Preparing the final array
        shocked_ret = self.ul_hist[self.close_ret_col][:49].tolist()

        # Setting a pointer for the loop
        p1 = 49

        while p1 < len(self.ul_hist[self.close_ret_col]):
            # Poisson random jumps
            jump = np.random.poisson(4)
            if jump > 4 + 2 * sqrt(4):
                # Normal magnitude for the shocks
                shock = np.random.normal(0, self.intensity * sqrt(sigma[p1]))
                shocked_ret.append(self.ul_hist[self.close_ret_col][p1] + shock)
            else:
                shock = np.random.normal(0, self.intensity * 0.1 * sqrt(sigma[p1]))
                shocked_ret.append(self.ul_hist[self.close_ret_col][p1] + shock)

            p1 += 1
        #
        # plt.figure()
        # plt.suptitle("Shock Process I: Implementation Example")
        # plt.subplot(1, 3, 1)
        # plt.plot(self.ul_hist[self.close_ret_col], color="dimgray")
        # plt.ylim(top=0.3, bottom=-0.4)
        # plt.title("Returns")
        # plt.subplot(1, 3, 2)
        # plt.plot(self.ul_hist.index, shocked_ret, color="skyblue")
        # plt.ylim(top=0.3, bottom=-0.4)
        # plt.title("Shocked Returns")
        # plt.subplot(1, 3, 3)
        # plt.plot(self.ul_hist.index, (shocked_ret/self.ul_hist[self.close_ret_col]), color="salmon")
        # plt.title("Ratio")
        # plt.show()

        return shocked_ret

    def norm_cluster_shock(self):
        """

        :param p
        :param intensity:
        :return:

        """
        # mean = np.mean(self.ul_hist[self.close_ret_col])
        sigma = self.ul_hist[self.close_ret_col].rolling(50).std()
        mu = self.ul_hist[self.close_ret_col].rolling(50).mean()
        shocked_ret = self.ul_hist[self.close_ret_col][:49].tolist()

        p1 = 49

        while p1 < len(self.ul_hist[self.close_ret_col]):
            jump = np.random.poisson(4)
            if jump > 4 + 2 * sqrt(4):
                shock = np.random.normal(mu[p1] / 0.02, self.intensity * sqrt(sigma[p1]))
                shocked_ret.append(self.ul_hist[self.close_ret_col][p1] + shock)
            else:
                shock = np.random.normal(mu[p1] / 0.02, self.intensity * 0.05 * sqrt(sigma[p1]))
                shocked_ret.append(self.ul_hist[self.close_ret_col][p1] + shock)

            p1 += 1
        #
        # plt.figure()
        # plt.suptitle("Shock Process II: Implementation Example")
        # plt.subplot(1, 3, 1)
        # plt.plot(self.ul_hist[self.close_ret_col], color="dimgray")
        # plt.ylim(top=0.6, bottom=-0.6)
        # plt.title("Returns")
        # plt.subplot(1, 3, 2)
        # plt.plot(self.ul_hist.index, shocked_ret, color="coral")
        # plt.ylim(top=0.6, bottom=-0.6)
        # plt.title("Shocked Returns")
        # plt.subplot(1, 3, 3)
        # plt.plot(self.ul_hist.index, (shocked_ret / self.ul_hist[self.close_ret_col]), color="salmon")
        # plt.title("Ratio")
        # plt.show()

        return shocked_ret

    def strat_return(self, exposure):
        """

        :return:

        """
        ret = [(a * b + (1 - a) * self.risk_free/252)
               for a, b in zip(exposure, self.ul_hist[self.close_ret_col])]

        # print(ret)
        # plt.plot(ret, color="orange")
        # plt.show()
        # plt.plot(self.ul_hist[self.close_ret_col], color = "green")
        # plt.show()
        return ret

    def strat_vol(self, exposure, n):
        """

        :return:

        """
        strat_returns = self.strat_return(exposure)
        vol = []

        for v in range(0, len(strat_returns)):
            if v > n - 1:
                vol.append(sqrt(1 / n * sum([i ** 2 for i in strat_returns[v - (n - 1):v]]) * 252))
            else:
                vol.append(sqrt(1 / (v + 1) * sum([i ** 2 for i in strat_returns[:(v + 1)]]) * 252))

        # print(vol)
        # plt.plot(vol, color="blue")
        # plt.show()

        return vol

    # def shock_simple_real_vol(self):
    #     """
    #
    #     :param intensity:
    #     :return:
    #
    #     """
    #     shocked_ret = []
    #     for v in self.norm_shock():
    #         shocked_ret.append(sqrt(v ** 2 * 252))
    #
    #     return shocked_ret
    #
    #     pass
    #
    # def cluster_shock_simple_real_vol(self):
    #     """
    #
    #     :param p:
    #     :param intensity:
    #     :return:
    #
    #     """
    #     shocked_ret = []
    #     for v in self.norm_cluster_shock():
    #         shocked_ret.append(sqrt(v ** 2 * 252))
    #
    #     return shocked_ret
    #
    # def shock_exposure(self):
    #     """
    #
    #     :param intensity:
    #     :return:
    #
    #     """
    #     exp = []
    #     for shocked_element in self.shock_simple_real_vol():
    #         exp.append(min(self.exp_cap, self.vt / shocked_element))
    #
    #     return exp
    #
    # def cluster_shock_exposure(self):
    #     """
    #
    #     :param p:
    #     :param intensity:
    #     :return:
    #
    #     """
    #
    #     exp = []
    #     for shocked_element in self.cluster_shock_simple_real_vol():
    #         exp.append(min(self.exp_cap, self.vt / shocked_element))
    #
    #     return exp

    # def shock_strat_return(self):
    #     """
    #
    #     :param intensity
    #     :return:
    #
    #     """
    #     ret = [(a*b + (1-a)*self.risk_free)
    #            for a, b in zip(self.shock_exposure(), self.ul_hist[self.close_ret_col])]
    #
    #     return ret
    #
    # def shock_strat_vol(self):
    #     """
    #
    #     :return:
    #
    #     """
    #     vol = [sqrt(r ** 2 * 252) for r in self.shock_strat_return()]
    #
    #     return vol
    #
    # def cluster_shock_strat_return(self):
    #     """
    #
    #     :param p:
    #     :param intensity:
    #     :return:
    #
    #     """
    #     ret = [(a*b + (1-a)*self.risk_free)
    #            for a, b in zip(self.cluster_shock_exposure(), self.ul_hist[self.close_ret_col])]
    #
    #     return ret
    #
    # def cluster_shock_strat_vol(self):
    #     """
    #
    #     :return:
    #
    #     """
    #     vol = [sqrt(r ** 2 * 252) for r in self.cluster_shock_strat_return()]
    #
    #     return vol

    """
    
    =====================================================================
    
                ARIMA & ARCH/GARCH, with specifications:
    
    =====================================================================
    
    """

    def check_heteroscedasticity(self):
        plt.plot(self.simple_real_var())
        plt.plot(np.mean(self.simple_real_var()))
        plt.show()

        return "Plots for HSD"

    def arima(self, p=5, q=0, d=2, start_pred=1000, end_pred=2000, print_resids=True):
        """

        :param p:
        :param q:
        :param d:
        :param start_pred:
        :param end_pred:
        :param print_resids:
        :return:

        """
        arima_model = ARIMA(self.ul_hist[self.close_ret_col], order=(p, q, d))
        arima_fit = arima_model.fit(disp=0)
        print(arima_fit.summary())

        if print_resids:
            residuals = pd.DataFrame(arima_fit.resid)
            residuals.plot()
            plt.show()
            residuals.plot(kind='kde')
            plt.show()
            print(residuals.describe())

        arima_prediction = ARIMAResults.predict(arima_fit, start=start_pred, end=end_pred)
        print(arima_prediction)

    def arch_plus(self, vol_model="ARCH", mean_model="ARX", p=1, q=1, h=100):
        """

        :param vol_model:
        :param mean_model:
        :param p:
        :param q:
        :param h:
        :return:

        """
        arima_model = arch_model(self.ul_hist[self.close_ret_col], mean=mean_model, vol=vol_model, p=p, q=q)
        arch_fit = arima_model.fit()
        print(arch_fit.summary())

        arch_yhat = arch_fit.forecast(horizon=h)
        arch_fit.plot()

        plt.plot(arch_yhat.variance.values[-1, :])
        plt.plot(self.simple_real_var()[-h::])
        plt.show()
