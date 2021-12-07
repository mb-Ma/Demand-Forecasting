import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from scipy import linalg
import datetime


class ReplenishUnit:
    def __init__(self,
                 unit,
                 demand_hist,
                 intransit,
                 qty_replenish,
                 qty_inventory_today,
                 qty_using_today,
                 arrival_sum,
                 lead_time
                 ):
        '''
        记录各补货单元状态
        :param unit:
        :param demand_hist: 净需求历史
        :param intransit: 补货在途
        :param qty_replenish: 补货记录
        :param qty_inventory_today: 当前可用库存
        :param qty_using_today: 当前已用库存（使用量）
        :param arrival_sum: 补货累计到达
        :param lead_time: 补货时长，交货时间
        '''
        self.unit = unit
        self.demand_hist = demand_hist
        self.intransit = intransit
        self.qty_replenish = qty_replenish
        self.qty_inventory_today = qty_inventory_today
        self.qty_using_today = qty_using_today
        self.arrival_sum = arrival_sum
        self.qty_using_today = qty_using_today
        self.lead_time = lead_time

    def update(self,
               date,
               arrival_today,
               demand_today):
        '''
        每日根据当天补货到达与当日净需求更新状态
        :param date:
        :param arrival_today: 当天补货到达
        :param demand_today: 当天净需求
        :return:
        '''
        self.qty_inventory_today += arrival_today # 本日的库存+本日的到货
        self.arrival_sum += arrival_today
        inv_today = self.qty_inventory_today
        if demand_today < 0: #　今天没有需求, 更新当前可用库存，　当前已用库存或当前需求量，取负表示正，这样当前可用库存是增加的．
            self.qty_inventory_today = self.qty_inventory_today + min(-demand_today, self.qty_using_today) # 如果今天释放量大于昨天的使用量，那就用昨天使用量
        else:
            self.qty_inventory_today = max(self.qty_inventory_today - demand_today, 0.0) #　求可用库存，当前库存减去需求量.
        self.qty_using_today = max(self.qty_using_today + min(demand_today, inv_today), 0.0) # good
        self.demand_hist = self.demand_hist.append({"ts": date, "unit": self.unit, "qty": demand_today}, ignore_index = True)

    def forecast_std(self, demand_hist):
        n = 3
        try:
            series = pd.Series(pd.Series(np.array(demand_hist["qty"].values[- 3 * 7:])).rolling(window=n).std().values[n-1:])
            model = ARIMA(series, order=(7, 0, 0))
            model_fit = model.fit()
            output = model_fit.forecast(steps=14)
            return output.values
        except:
            std_averge = np.std(self.demand_hist["qty"].values[-3 * self.lead_time:])
            return np.array([std_averge] * self.lead_time)

    def forecast_function(self,
                          demand_hist):
        # s1
        # demand_average = np.mean(self.demand_hist["qty"].values[-3 * self.lead_time:])
        # return [demand_average] * 90
        # s2
        try:
            series = pd.Series(np.array(self.demand_hist["qty"].values[- 3 * 7:]))
            model = ARIMA(series, order=(7, 0, 0))
            model_fit = model.fit()
            output = model_fit.forecast(steps=14)
            return output.values
        except:
            demand_average = np.mean(self.demand_hist["qty"].values[-3 * self.lead_time:])
            return [demand_average] * self.lead_time

        # s3
        # try:
        #     series = pd.Series(np.array(self.demand_hist["qty"].values[- 3 * 7:]))
        #     mod = sm.tsa.statespace.SARIMAX(series, order=(0, 0, 0), seasonal_order=(0, 1, 1, 14), enforce_stationarity=False,
        #                                     enforce_invertibility=False)
        #     results = mod.fit()
        #     return results.get_forecast(steps=14).predicted_mean
        # except:
        #     demand_average = np.mean(self.demand_hist["qty"].values[-3 * self.lead_time:])
        #     return [demand_average] * 90

    def replenish_function(self,
                           date):
        '''
        根据当前状态判断需要多少的补货量
        补货的策略由选手决定，这里只给一个思路
        :param date:
        :return:
        '''
        replenish = 0.0
        if date.dayofweek != 0:
            #周一为补货决策日，非周一不做决策
            pass
        else:
            #预测未来需求量
            qty_demand_forecast = self.forecast_function(demand_hist = self.demand_hist)

            # 预测未来需求量的方差
            qty_demand_std_forecast = self.forecast_std(demand_hist = self.demand_hist)

            #计算在途的补货量
            qty_intransit = sum(self.intransit) - self.arrival_sum

            # 安全库存 用来抵御需求的波动性 选手可以换成自己的策略
            # 1. safety_stock = (max(self.demand_hist["qty"].values[-3 * self.lead_time:]) - (np.mean(self.demand_hist["qty"].values[- 3 * self.lead_time:]))) * self.lead_time
            # 2. safety_stock = (max(qty_demand_forecast) - (np.mean(qty_demand_forecast))) * self.lead_time
            # safety_stock = np.sum(max(qty_demand_forecast) - qty_demand_forecast) # 目前最优表现
            try:
                safety_stock = np.sum(max(qty_demand_forecast) - qty_demand_forecast + 1.65 * qty_demand_std_forecast * qty_demand_forecast)
            except:
                print(qty_demand_std_forecast)
                exit(0)
            #再补货点，用来判断是否需要补货 选手可以换成自己的策略
            reorder_point = sum(qty_demand_forecast[:self.lead_time]) + safety_stock

            # # mean 方法的求值，然后将二者做一个平均
            # qty_demand_forecast = [np.mean(self.demand_hist["qty"].values[-3 * self.lead_time:])] * self.lead_time
            # safety_stock = (max(self.demand_hist["qty"].values[-3 * self.lead_time:]) - (np.mean(self.demand_hist["qty"].values[- 3 * self.lead_time:]))) * self.lead_time
            # reorder_point2 = sum(qty_demand_forecast) + safety_stock
            #
            # reorder_point = (reorder_point2 + reorder_point) / 2.0

            #判断是否需要补货并计算补货量，选手可以换成自己的策略，可以参考赛题给的相关链接
            if self.qty_inventory_today + qty_intransit < reorder_point:
                replenish = reorder_point - (self.qty_inventory_today + qty_intransit)

            self.qty_replenish.at[date] = replenish
            self.intransit.at[date + self.lead_time * date.freq] = replenish


class SupplyChainRound1Baseline:
    def __init__(self):
        self.using_hist = pd.read_csv("../data/Round_B/demand_train_B.csv") # 历史需求
        self.using_future = pd.read_csv("../data/Round_B/demand_test_B.csv") # 未来信息
        self.inventory = pd.read_csv("../data/Round_B/inventory_info_B.csv")
        self.last_dt = pd.to_datetime("20210301")
        self.start_dt = pd.to_datetime("20210302")
        self.end_dt = pd.to_datetime("20210607")
        self.lead_time = 14

    def run(self):
        self.using_hist["ts"] = self.using_hist["ts"].apply(lambda x:pd.to_datetime(x))
        self.using_future["ts"] = self.using_future["ts"].apply(lambda x:pd.to_datetime(x))
        qty_using = pd.concat([self.using_hist, self.using_future])
        date_list = pd.date_range(start = self.start_dt, end = self.end_dt)
        unit_list = self.using_future["unit"].unique()
        res = pd.DataFrame(columns = ["unit", "ts", "qty"])

        replenishUnit_dict = {} # 补充量
        demand_dict = {}

        #初始化，记录各补货单元在评估开始前的状态
        for chunk in qty_using.groupby("unit"): # 按unit分组计算
            unit = chunk[0]
            demand = chunk[1]
            demand.sort_values("ts", inplace = True, ascending = True)

            #计算净需求量 = 毛需求－(现有库存－现有库存已分配量)－在途量＋安全库存
            demand["diff"] = demand["qty"].diff().values
            demand["qty"] = demand["diff"]
            del demand["diff"]
            demand = demand[1:] # 去掉第一个为nan的值  replenish 补货量
            replenishUnit_dict[unit] = ReplenishUnit(unit = unit,
                                                     demand_hist = demand[demand["ts"] < self.start_dt],
                                                     intransit = pd.Series(index = date_list.tolist(), data = [0.0] * (len(date_list))),
                                                     qty_replenish = pd.Series(index = date_list.tolist(), data = [0.0] * (len(date_list))),
                                                     qty_inventory_today = self.inventory[self.inventory["unit"] == unit]["qty"].values[0],
                                                     qty_using_today = self.using_hist[(self.using_hist["ts"] == self.last_dt) & (self.using_hist["unit"] == unit)]["qty"].values[0],
                                                     arrival_sum = 0.0,
                                                     lead_time = self.lead_time)

            #记录评估周期内的净需求量
            demand_dict[unit] = demand[(demand["unit"] == unit) & (demand["ts"] >= self.start_dt)]

        for date in date_list:
            #按每日净需求与每日补货到达更新状态，并判断补货量
            for unit in unit_list:
                demand = demand_dict[unit]
                demand_today = demand[demand["ts"] == date]["qty"].values[0]
                arrival = replenishUnit_dict[unit].intransit.get(date, default = 0.0)
                replenishUnit_dict[unit].update(date = date,
                                                arrival_today = arrival,
                                                demand_today = demand_today)
                replenishUnit_dict[unit].replenish_function(date)

        for unit in unit_list:
            res_unit = replenishUnit_dict[unit].qty_replenish
            res_unit = pd.DataFrame({"unit": unit,
                                     "ts": res_unit.index,
                                     "qty": res_unit.values})
            res_unit = res_unit[res_unit["ts"].apply(lambda x:x.dayofweek == 0)]
            res = pd.concat([res, res_unit])
        #输出结果
        res.to_csv("baseline.csv")


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    supplyChainRound1Baseline = SupplyChainRound1Baseline()
    supplyChainRound1Baseline.run()
    end_time = datetime.datetime.now()
    print('Cost time is {}s'.format((end_time-start_time).seconds))
