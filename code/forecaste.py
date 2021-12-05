from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def extract_hist_demand():
    using_hist = pd.read_csv("../data/Dataset/demand_train_A.csv")  # 历史需求
    using_future = pd.read_csv("../data/Dataset/demand_test_A.csv")  # 未来信息
    inventory = pd.read_csv("../data/Dataset/inventory_info_A.csv")
    last_dt = pd.to_datetime("20210301")
    start_dt = pd.to_datetime("20210302")
    end_dt = pd.to_datetime("20210607")
    lead_time = 14

    using_hist["ts"] = using_hist["ts"].apply(lambda x: pd.to_datetime(x))
    using_future["ts"] = using_future["ts"].apply(lambda x: pd.to_datetime(x))
    qty_using = pd.concat([using_hist, using_future])
    date_list = pd.date_range(start=start_dt, end=end_dt)
    unit_list = using_future["unit"].unique()
    res = pd.DataFrame(columns=["unit", "ts", "qty"])

    replenishUnit_dict = {}  # 补充量
    demand_dict = {}
    for chunk in qty_using.groupby("unit"):  # 按unit分组计算
        unit = chunk[0]
        demand = chunk[1]
        demand.sort_values("ts", inplace=True, ascending=True)

        # 计算净需求量 = 毛需求－(现有库存－现有库存已分配量)－在途量＋安全库存
        demand["diff"] = demand["qty"].diff().values
        demand["qty"] = demand["diff"]
        del demand["diff"]
        demand = demand[1:]

# tmp_data = [ -2.42513021,   6.0546875 ,  -1.42686632,  21.02322049, 14.12217882,  11.52886285,  19.18945312,  20.31792535,
#          0.53710938,   5.3765191 ,   9.73849826,  11.5234375 , 9.16883681,  -8.30078125,  13.65559896,   1.171875  ,
#          1.14474826,   0.48828125,   4.63867188,   0.5859375 ,
#         -3.27148438,   0.65104167,   1.32921007,  37.33181424,
#        179.35655382,   4.07986111,   4.31857639,  32.16145833,
#         55.60438368,   6.33138021,  30.81054688,   3.10329861,
#        -13.82378472,   3.29318576,   6.21744792,  11.64279514,
#         15.47309028, -13.13476562,   2.89171007,   1.8608941 ,
#          3.46679688,   2.66384549]

# autocorrelation_plot(tmp_data)
# pyplot.show()

series = pd.Series(np.array(tmp_data))
# series.index = series.index.to_period('D')

model = ARIMA(series, order=(14, 1, 0))
model_fit = model.fit()

output = model_fit.forecast(steps=14)

print(output.values)