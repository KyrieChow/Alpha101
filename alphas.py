import numpy as np
import pandas as pd
import copy
from scipy.stats import rankdata
from utils import *


class Alphas:
    def __init__(self, data):
        """

        :param data: multi-index pd.DataFrame
        """
        self.open = data['OpenPrice']
        self.high = data['HighPrice']
        self.low = data['LowPrice']
        self.close = data['ClosePrice']
        self.volume = data['Volume']
        self.returns = data['ClosePrice'].pct_change()
        self.vwap = data['VWAP']
        self.cap = data['MktCap']

    def __methods__(self):
        """
        get all methods except internal methods
        :return:
        """
        return list(filter(
            lambda m: not m.startswith("__") and not m.endswith("__") and callable(getattr(self, m)), dir(self)))

    def alpha001(self):
        """
        Alpha#1	 (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
        :return: pd.DataFrame
        """
        inner = copy.deepcopy(self.close)
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(pow(inner, 2), 5)) - 0.5

    def alpha002(self):
        """
        Alpha#2	 (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        :return:
        """
        return (-1 * correlation(
            rank(delta(np.log(self.volume), 2)), rank((self.close - self.open) / self.open),
            6)).replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha003(self):
        """
        Alpha#3	 (-1 * correlation(rank(open), rank(volume), 10))
        :return:
        """

        return (-1 * correlation(rank(self.open), rank(self.volume),
                                 10)).replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha004(self):
        """
        Alpha#4	 (-1 * Ts_Rank(rank(low), 9))
        :return:
        """
        return -1 * ts_rank(rank(self.low), 9)

    def alpha005(self):
        """
        Alpha#5	 (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
        :return:
        """
        return rank((self.open - (ts_sum(self.vwap, 10) / 10))) * (-1 * np.abs(rank((self.close - self.vwap))))

    def alpha006(self):
        """
        Alpha#6	 (-1 * correlation(open, volume, 10))
        :return:
        """

        return -1 * correlation(self.open, self.volume,
                                10).replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha007(self):
        """
        Alpha#7	 ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
        :return:
        """
        alpha = -1 * ts_rank(np.abs(delta(self.close, 7)), 60) * np.sign(delta(self.close, 7))
        alpha[sma(self.volume, 20) >= self.volume] = -1
        return alpha

    def alpha008(self):
        """
        Alpha#8	 (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
        :return:
        """
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))

    def alpha009(self):
        """
        Alpha#9	 ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
        ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
        :return:
        """
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    def alpha010(self):
        """
        Alpha#10	 rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
        ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
        :return:
        """
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    def alpha011(self):
        """
        Alpha#11	 ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
        :return:
        """
        return ((rank(ts_max((self.vwap - self.close), 3)) +
                 rank(ts_min((self.vwap - self.close), 3))) *
                rank(delta(self.volume, 3)))

    def alpha012(self):
        """
        Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        :return:
        """
        return np.sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self):
        """
        Alpha#13	 (-1 * rank(covariance(rank(close), rank(volume), 5)))
        :return:
        """
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self):
        """
        Alpha#14	 ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        :return:
        """
        return -1 * rank(delta(self.returns, 3)) * correlation(
            self.open, self.volume, 10).replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha015(self):
        """
        Alpha#15	 (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        :return:
        """
        return -1 * ts_sum(rank(
            correlation(rank(self.high), rank(self.volume), 3).replace([-np.inf, np.inf], 0).fillna(value=0)), 3)

    def alpha016(self):
        """
        Alpha#16	 (-1 * rank(covariance(rank(high), rank(volume), 5)))
        :return:
        """
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self):
        """
        Alpha#17	 (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)))
        *rank(ts_rank((volume / adv20), 5)))
        :return:
        """

        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(delta(delta(self.close, 1), 1)) *
                     rank(ts_rank((self.volume / sma(self.volume, 20)), 5)))

    def alpha018(self):
        """
        Alpha#18	 (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
        :return:
        """

        return -1 * (rank((stddev(np.abs((self.close - self.open)), 5) + (self.close - self.open)) +
                          correlation(self.close, self.open, 10).replace([-np.inf, np.inf], 0).fillna(value=0)))

    def alpha019(self):
        """
        Alpha#19	 ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
        :return:
        """
        return ((-1 * np.sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250))))

    def alpha020(self):
        """
        Alpha#20	 (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1))))
        * rank((open -delay(low, 1))))
        :return:
        """
        return -1 * (rank(self.open - delay(self.high, 1)) *
                     rank(self.open - delay(self.close, 1)) *
                     rank(self.open - delay(self.low, 1)))

    def alpha021(self):
        """
        Alpha#21	 ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ?
         (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ?
          1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
        :return:
        """
        cond_1 = ts_sum(self.close, 8) / 8 + stddev(self.close, 8) < ts_sum(self.close, 2) / 2
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index)
        alpha[cond_1 | cond_2] = -1
        return alpha

    def alpha022(self):
        """
        Alpha#22	 (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
        :return:
        """
        return -1 * delta(
            correlation(self.high, self.volume, 5).replace([-np.inf, np.inf], 0).fillna(value=0),
            5) * rank(stddev(self.close, 20))

    def alpha023(self):
        """
        Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        :return:
        """
        cond = ts_sum(self.high, 20)/20 < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index)
        alpha[cond] = -1 * delta(self.high, 2).fillna(value=0)
        return alpha

    def alpha024(self):
        """
        Alpha#24	 ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05)
        ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ?
        (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
        :return:
        """
        cond = delta(ts_sum(self.close, 100)/100, 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha

    def alpha025(self):
        """
        Alpha#25	 rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
        :return:
        """
        return rank(((((-1 * self.returns) *
                       sma(self.volume, 20)
                       ) * self.vwap) * (self.high - self.close)))

    def alpha026(self):
        """
        Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        :return:
        """
        return -1 * ts_max(
            correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5).replace(
                [-np.inf, np.inf], 0).fillna(value=0), 3)

    def alpha027(self):
        """
        Alpha#27	 ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
        :return:
        """
        cond = rank((ts_sum(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0)) > 0.5
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index)
        alpha[cond] = -1
        return alpha

    def alpha028(self):
        """
        Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        :return:
        """
        return scale(((correlation(sma(self.volume, 20), self.low, 5).replace(
            [-np.inf, np.inf], 0).fillna(value=0) + ((self.high + self.low) / 2)) - self.close))

    def alpha029(self):
        """
        Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((
        -1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
        :return:
        """
        return (pd.DataFrame(np.min(product(rank(rank(scale(np.log(ts_sum(ts_min(rank(rank(
            (-1 * rank(delta((self.close - 1), 5))))), 2), 1))))), 1), 5, axis=1),
            index=self.close.index) + ts_rank(delay((-1 * self.returns), 6), 5))

    def alpha030(self):
        """
        Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) +
         sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) *
          sum(volume, 5)) / sum(volume, 20))
        :return:
        """
        return ((1.0 - rank(((np.sign((self.close - delay(self.close, 1))) + np.sign(
            (delay(self.close, 1) - delay(self.close, 2)))) + np.sign(
            (delay(self.close, 2) - delay(self.close, 3)))))) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    def alpha031(self):
        """
        Alpha#31	 ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) +
         rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
        :return:
        """
        return ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))), 10)))) +
                 rank((-1 * delta(self.close, 3)))) + np.sign(
            scale(correlation(sma(self.volume, 20), self.low, 12).replace([-np.inf, np.inf], 0).fillna(value=0))))

    def alpha032(self):
        """
        Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
        :return:
        """
        return scale(((ts_sum(self.close, 7) / 7) - self.close)) + (
                20 * scale(correlation(self.vwap, delay(self.close, 5), 230)))

    def alpha033(self):
        """
        Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
        :return:
        """
        return rank(-1 + (self.open / self.close))

    def alpha034(self):
        """
        Alpha#34	 rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        :return:
        """
        return rank(((1 - rank((stddev(self.returns, 2) / stddev(self.returns, 5)).replace(
            [-np.inf, np.inf], 1).fillna(value=1))) + (1 - rank(delta(self.close, 1)))))

    def alpha035(self):
        """
        Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
        :return:
        """
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))

    def alpha036(self):
        """
        Alpha#36	 (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) +
         (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) +
          rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
        :return:
        """
        return (((((2.21 * rank(correlation((self.close - self.open), delay(self.volume, 1), 15))) + (
                0.7 * rank((self.open - self.close)))) + (
                          0.73 * rank(ts_rank(delay((-1 * self.returns), 6), 5)))) + rank(
            np.abs(correlation(self.vwap, sma(self.volume, 20), 6)))) + (
                        0.6 * rank((((ts_sum(self.close, 200) / 200) - self.open) * (self.close - self.open)))))

    def alpha037(self):
        """
        Alpha#37	 (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        :return:
        """
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    def alpha038(self):
        """
        Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        :return:
        """
        return -1 * rank(ts_rank(self.close, 10)) * rank(
            (self.close / self.open).replace([-np.inf, np.inf], 1).fillna(value=1))

    def alpha039(self):
        """
        Alpha#39	 ((-1 * rank((delta(close, 7) *
        (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
        :return:
        """
        return ((-1 * rank(delta(self.close, 7) *
                           (1 - rank(decay_linear(self.volume / sma(self.volume, 20), 9))))) *
                (1 + rank(ts_sum(self.returns, 250))))

    def alpha040(self):
        """
        Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
        :return:
        """
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def alpha041(self):
        """
        Alpha#41	 (((high * low)^0.5) - vwap)
        :return:
        """
        return (self.high * self.low) ** 0.5 - self.vwap

    def alpha042(self):
        """
        Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
        :return:
        """
        return rank((self.vwap - self.close)) / rank((self.vwap + self.close))

    def alpha043(self):
        """
        Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        :return:
        """
        return ts_rank(self.volume / sma(self.volume, 20), 20) * ts_rank((-1 * delta(self.close, 7)), 8)

    def alpha044(self):
        """
        Alpha#44	 (-1 * correlation(high, rank(volume), 5))
        :return:
        """
        return -1 * correlation(self.high, rank(self.volume), 5).replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha045(self):
        """
        Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) *
        correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
        :return:
        """
        return -1 * (rank(ts_sum(delay(self.close, 5), 20) / 20) *
                     correlation(self.close, self.volume, 2).replace([-np.inf, np.inf], 0).fillna(value=0) *
                     rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))

    def alpha046(self):
        """
        Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) -
        ((delay(close, 10) - close) / 10))) ?(-1 * 1) :
        (((((delay(close, 20) - delay(close, 10)) / 10) -
        ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
        :return:
        """
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = -1 * (self.close - delay(self.close, 1))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    def alpha047(self):
        """
        Alpha#47	 ((((rank((1 / close)) * volume) / adv20) *
        ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
        :return:
        """
        return ((((rank((1 / self.close)) * self.volume) / sma(self.volume, 20)) * (
                (self.high * rank((self.high - self.close))) / (ts_sum(self.high, 5) / 5))) - rank(
            (self.vwap - delay(self.vwap, 5))))

    # Alpha#48	 (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *
    # delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))

    def alpha049(self):
        """
        Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) -
        ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        :return:
        """
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = -1 * (self.close - delay(self.close, 1))
        alpha[inner < -0.1] = 1
        return alpha

    def alpha050(self):
        """
        Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        :return:
        """
        return -1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha051(self):
        """
        Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) -
        ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        :return:
        """
        inner = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        alpha = -1 * (self.close - delay(self.close, 1))
        alpha[inner < -0.05] = 1
        return alpha

    def alpha052(self):
        """
        Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
        rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
        :return:
        """
        return (((-1 * ts_min(self.low, 5)) + delay(ts_min(self.low, 5), 5)) *
                rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume, 5)

    def alpha053(self):
        """
        Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        :return:
        """
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) /
                           (self.close - self.low).replace(0, 0.0001)), 9)

    def alpha054(self):
        """
        Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        :return:
        """
        return -1 * (self.low - self.close) * (self.open ** 5) / (
                (self.low - self.high).replace(0, -0.0001) * (self.close ** 5))

    def alpha055(self):
        """
        Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) /
         (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
        :return:
        """
        return -1 * correlation(rank(
            (self.close - ts_min(self.low, 12)) / (
                (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
            )), rank(self.volume), 6).replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha056(self):
        """
        Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
        :return:
        """
        return -rank((ts_sum(self.returns, 10) / ts_sum(ts_sum(self.returns, 2), 3))) * rank((self.returns * self.cap))

    def alpha057(self):
        """
        Alpha#57	 (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
        :return:
        """
        return -pd.DataFrame(self.close - self.vwap,
                             index=self.close.index) / decay_linear(rank(ts_argmax(self.close, 30)), 2)

    # Alpha#58	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))

    # Alpha#59	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))

    def alpha060(self):
        """
        Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) /
        (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
        :return:
        """
        return - ((2 * scale(rank(
            ((self.close - self.low) - (self.high - self.close)) * self.volume /
            (self.high - self.low).replace(0, 0.0001)
        ))) - scale(rank(ts_argmax(self.close, 10))))

    def alpha061(self):
        """
        Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))

        data type issue, convert to int for some params
        :return:
        """
        return rank((self.vwap - ts_min(self.vwap, 16))) < rank(
            correlation(self.vwap, sma(self.volume, 180), 18))

    def alpha062(self):
        """
        Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) <
        rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)

        data type issue, convert to int for some params
        :return:
        """
        return -(rank(correlation(self.vwap, ts_sum(sma(self.volume, 20), 22), 10)) < rank(
            ((rank(self.open) * 2) < (rank(((self.high + self.low) / 2)) + rank(self.high)))))

    # Alpha#63	 ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)

    def alpha064(self):
        """
        Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),
        sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) +
        (vwap * (1 -0.178404))), 3.69741))) * -1)

        data type issue, convert to int for some params
        :return:
        """
        return ((rank(correlation(ts_sum(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13),
                                  ts_sum(sma(self.volume, 120), 13), 17)) < rank(
            delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))), 4))) * -1)

    def alpha065(self):
        """
        Alpha#65	 ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))),
        sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
        :return:
        """
        return ((rank(correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))),
                                  ts_sum(sma(self.volume, 60), 9), 6)) < rank(
            (self.open - ts_min(self.open, 14)))) * -1)

    def alpha066(self):
        """
        Alpha#66	 ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) +
         (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
        :return:
        """
        return ((rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(
            decay_linear(((((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (
                    self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)

    # Alpha#67	 ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)

    def alpha068(self):
        """
        Alpha#68	 ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <
        rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
        :return:
        """
        return ((ts_rank(correlation(rank(self.high), rank(sma(self.volume, 15)), 9), 14) < rank(
            delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1))) * -1)

    # Alpha#69	 ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)

    # Alpha#70	 ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)

    def alpha071(self):
        """
        Alpha#71	 max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976),
         Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948),
         Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
        :return:
        """
        return pd.DataFrame(np.max(ts_rank(decay_linear(
            correlation(ts_rank(self.close, 3), ts_rank(sma(self.volume, 180), 12), 18), 4), 16), ts_rank(
            decay_linear((rank(((self.low + self.open) - (self.vwap + self.vwap))) ** 2), 16), 4), axis=1),
            index=self.close.index)

    def alpha072(self):
        """
        Alpha#72	 (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /
        rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
        :return:
        """
        return (rank(decay_linear(correlation(((self.high + self.low) / 2), sma(self.volume, 40), 9), 10)) / rank(
            decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3)))

    def alpha073(self):
        """
        Alpha#73	 (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
        Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) /
         ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
        :return:
        """
        return pd.DataFrame(np.max(
            rank(decay_linear(delta(self.vwap, 5), 3)), ts_rank(decay_linear(
                ((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) /
                  ((self.open * 0.147155) + (self.low * (1 - 0.147155)))) * -1), 3), 17), axis=1) * -1,
                            index=self.close.index)

    def alpha074(self):
        """
        Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <
        rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
        :return:
        """
        return ((rank(correlation(self.close, ts_sum(sma(self.volume, 30), 37), 15)) < rank(
            correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11))) * -1)

    def alpha075(self):
        """
        Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
        :return:
        """
        return rank(correlation(self.vwap, self.volume, 4)) < rank(
            correlation(rank(self.low), rank(sma(self.volume, 50)), 12))

    # Alpha#76	 (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)

    def alpha077(self):
        """
        Alpha#77	 min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)),
         20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
        :return:
        """
        return pd.DataFrame(np.min(rank(decay_linear(
            ((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20)), rank(
            decay_linear(correlation(((self.high + self.low) / 2), sma(self.volume, 40), 3), 6)), axis=1),
            index=self.close.index)

    def alpha078(self):
        """
        Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))),
        19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
        :return:
        """
        return (rank(correlation(
            ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20), ts_sum(
                sma(self.volume, 40), 20), 7)).pow(rank(correlation(rank(self.vwap), rank(self.volume), 6))))

    # Alpha#79	 (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))

    # Alpha#80	 ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)

    def alpha081(self):
        """
        Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap,
        sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
        :return:
        """
        return ((rank(np.log(product(rank((
                rank(correlation(self.vwap, ts_sum(sma(self.volume, 10), 50), 8)) ** 4)), 15))) < rank(
            correlation(rank(self.vwap), rank(self.volume), 5))) * -1)

    # Alpha#82	 (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)

    def alpha083(self):
        """
        Alpha#83	 ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) *
         rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
        :return:
        """
        return ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) / (
                ((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))

    def alpha084(self):
        """
        Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
        :return:
        """
        return pow(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close, 5))

    def alpha085(self):
        """
        Alpha#85	 (rank(correlation(((high * 0.876703) +
        (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(
        Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
        :return:
        """
        return (rank(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), sma(self.volume, 30),
                10)).pow(rank(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10), 7))))

    def alpha086(self):
        """
        Alpha#86	 ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) <
         rank(((open+ close) - (vwap + open)))) * -1)
        :return:
        """
        return ((ts_rank(correlation(self.close, ts_sum(sma(self.volume, 20), 15), 6), 20) < rank(
            ((self.open + self.close) - (self.vwap + self.open)))) * -1)

    # Alpha#87	 (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)

    def alpha088(self):
        """
        Alpha#88	 min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)),
         Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
        :return:
        """
        return pd.DataFrame(np.min(rank(decay_linear(((rank(self.open) + rank(self.low)) - (
                rank(self.high) + rank(self.close))), 8)), ts_rank(decay_linear(correlation(ts_rank(self.close, 8),
                ts_rank(sma(self.volume, 60), 21), 8), 7), 3), axis=1), index=self.close.index)

    # Alpha#89	 (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))

    # Alpha#90	 ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)

    # Alpha#91	 ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)

    def alpha092(self):
        """
        Alpha#92	 min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683),
         Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
        :return:
        """
        return pd.DataFrame(np.min(ts_rank(decay_linear(
            ((((self.high + self.low) / 2) + self.close) < (self.low + self.open)), 15), 19), ts_rank(
            decay_linear(correlation(rank(self.low), rank(sma(self.volume, 30)), 8), 7), 7), axis=1),
            index=self.close.index)

    # Alpha#93	 (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))

    def alpha094(self):
        """
        Alpha#94	 ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,
        19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
        :return:
        """
        return ((rank((self.vwap - ts_min(self.vwap, 12))).pow(
            ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(sma(self.volume, 60), 4), 18), 3)) * -1))

    def alpha095(self):
        """
        Alpha#95	 (rank((open - ts_min(open, 12.4105))) <
        Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
        :return:
        """
        return (rank((self.open - ts_min(self.open, 12))) < ts_rank(
            (rank(correlation(ts_sum(((self.high + self.low) / 2), 19),
                              ts_sum(sma(self.volume, 40), 19), 13))**5), 12))

    def alpha096(self):
        """
        Alpha#96	 (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151),
        Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),
        Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
        :return:
        """
        return pd.DataFrame(np.max(ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4), 8),
        ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7),
        ts_rank(sma(self.volume, 60), 4), 4), 13), 14), 13), axis=1) * -1, index=self.close.index)

    # Alpha#97	 ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)

    def alpha098(self):
        """
        Alpha#98	 (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -
        rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
        :return:
        """
        return (rank(decay_linear(correlation(self.vwap, ts_sum(sma(self.volume, 5), 26), 5), 7)) -
        rank(decay_linear(ts_rank(ts_argmin(correlation(rank(self.open), rank(sma(self.volume, 15)), 21), 9), 7), 8)))

    def alpha099(self):
        """
        Alpha#99	 ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <
        rank(correlation(low, volume, 6.28259))) * -1)
        :return:
        """
        return ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(
            sma(self.volume, 60), 20), 9)) < rank(correlation(self.low, self.volume, 6))) * -1)

    # Alpha#100	 (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),IndClass.subindustry))) * (volume / adv20))))

    def alpha101(self):
        """
        Alpha#101	 ((close - open) / ((high - low) + .001))
        :return:
        """
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


def get_alpha(df):
    """

    :param df:
    :return: Alphas 101
    """
    al = Alphas(df)
    factors = []
    for alpha in al.__methods__():
        try:
            fun = eval("al."+alpha)
            temp = fun()
            temp.columns = [alpha]
            factors.append(temp)
        except:
            print("Error in ", alpha)
    return factors
