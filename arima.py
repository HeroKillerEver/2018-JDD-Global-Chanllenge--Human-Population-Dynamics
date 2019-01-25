import datetime
import warnings

import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
import pickle
import time

warnings.filterwarnings('ignore')


##### Below predict the missing 5 days
flow_df = pd.read_csv('./DATA/flow_train.csv')
flow_df = flow_df.sort_values(by=['city_code', 'district_code', 'date_dt'])

date_dt = list()
init_date = datetime.date(2017, 8, 19)
for delta in range(5):
    _date = init_date + datetime.timedelta(days=delta)
    date_dt.append(_date.strftime('%Y%m%d'))

district_code_values = flow_df['district_code'].unique()
preds_df_1 = pd.DataFrame()
tmp_df_columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']

for i, district_code in enumerate(district_code_values):
    sub_df = flow_df[flow_df['district_code'] == district_code]
    
    # truncate the data before the missing five days
    sub_df = sub_df[sub_df['date_dt'] <= 20170818]
    
    city_code = sub_df['city_code'].iloc[0]

    predict_columns = ['dwell', 'flow_in', 'flow_out']
    tmp_df = pd.DataFrame(data=date_dt, columns=['date_dt'])
    tmp_df['city_code'] = city_code
    tmp_df['district_code'] = district_code

    for column in predict_columns:
        ts = np.array(sub_df[column])

        arima_model = auto_arima(ts, start_p=1, max_p=9, start_q=1, max_q=9, max_d=5,
                                 start_P=1, max_P=9, start_Q=1, max_Q=9, max_D=5,
                                 m=7, random_state=2018,
                                 trace=True,
                                 seasonal=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True, n_jobs=5)

        preds = arima_model.predict(n_periods=5)
        preds = pd.Series(preds)

        tmp_df = pd.concat([tmp_df, preds], axis=1)

    print("iteration {0} complete".format(i))
    
    tmp_df.columns = tmp_df_columns
    preds_df_1 = pd.concat([preds_df_1, tmp_df], axis=0, ignore_index=True)
    
    preds_df_1 = preds_df_1.sort_values(by=['date_dt'])
#     preds_df_1.to_csv('prediction_arima_fill_missing_5_days.csv', index=False, header=False)



##### Below predict the last 10 days
date_dt = list()
init_date = datetime.date(2017, 11, 5)
for delta in range(10):
    _date = init_date + datetime.timedelta(days=delta)
    date_dt.append(_date.strftime('%Y%m%d'))

district_code_values = flow_df['district_code'].unique()
preds_df_2 = pd.DataFrame()
tmp_df_columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']

for i, district_code in enumerate(district_code_values):
    sub_df = flow_df[flow_df['district_code'] == district_code]
    
    city_code = sub_df['city_code'].iloc[0]

    predict_columns = ['dwell', 'flow_in', 'flow_out']
    tmp_df = pd.DataFrame(data=date_dt, columns=['date_dt'])
    tmp_df['city_code'] = city_code
    tmp_df['district_code'] = district_code

    for column in predict_columns:
        ts = np.array(sub_df[column])

        arima_model = auto_arima(ts, start_p=1, max_p=9, start_q=1, max_q=9, max_d=5,
                                 start_P=1, max_P=9, start_Q=1, max_Q=9, max_D=5,
                                 m=7, random_state=2018,
                                 trace=True,
                                 seasonal=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True, n_jobs=5)

        preds = arima_model.predict(n_periods=10)        
        preds = pd.Series(preds)

        tmp_df = pd.concat([tmp_df, preds], axis=1)

    print("iteration {0} complete".format(i))
    
    tmp_df.columns = tmp_df_columns
    preds_df_2 = pd.concat([preds_df_2, tmp_df], axis=0, ignore_index=True)
    
    preds_df_2 = preds_df_2.sort_values(by=['date_dt'])
#     preds_df_2.to_csv('prediction_arima_pred_last_10_days.csv', index=False, header=False)


predict = pd.concat([preds_df_1, preds_df_2])
predict.to_csv('predictionARIMA.csv', index=False, header=False) # 0.2487

