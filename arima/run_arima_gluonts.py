# -*- coding: UTF-8 -*-
# author : joelonglin
import pandas as pd
import os
import sys
#将当前工作路径添加进去
sys.path.insert(0, '.')
from gluonts.model.r_forecast import RForecastPredictor
from data_process.utils import create_dataset_if_not_exist , add_time_mark_to_file
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

print('所有导包都成功...')

target = 'Aluminium,Copper,Lead,Nickel,Tin,Zinc'
start = '2014-01-02'
freq = '1B'
timestep = 1303
pred = 1
past = 90
slice = 'overlap'
ds_name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'

result_params = '_'.join([
                                'freq(%s)'%(freq),
                               'past(%d)'%(past)
                                ,'pred(%d)'%(pred)
                             ]
                        )

if __name__ == '__main__':
    # 导入 target 的数据
    if slice == 'overlap':
        series = timestep - past - pred + 1
        print('每个数据集的序列数量为 ', series)
    elif slice == 'nolap':
        series = timestep // (past + pred)
        print('每个数据集的序列数量为 ', series)
    else:
        series = 1
        print('每个数据集的序列数量为 ', series ,'情景为长单序列')
    result_root_path  = 'evaluate/results/{}_slice({})_past({})_pred({})'.format(target.replace(',' ,'_') , slice ,past , pred)
    if not os.path.exists(result_root_path):
        os.makedirs(result_root_path)
    forecast_result_saved_path = os.path.join(result_root_path,'prophet(%s)_' % (target.replace(',' ,'_')) + result_params + '.pkl')
    forecast_result_saved_path = add_time_mark_to_file(forecast_result_saved_path)

    # 目标序列的数据路径
    target_path = {ds_name: ds_name_prefix.format(
            '%s_start(%s)_freq(%s)'%(ds_name, start,freq), 'steps(%d)_slice(%s)_DsSeries(%d)'%(timestep , slice, series),
            'train(%d)'%past, 'pred(%d)'%pred,
    ) for ds_name in target.split(',')}

    create_dataset_if_not_exist(
        paths=target_path, start=start, past_length=past
        , pred_length=pred, slice=slice
        , timestep=timestep, freq=freq
    )


    if not os.path.exists(forecast_result_saved_path):
        sample_forecasts = []
        
        for target_name in target_path:
            target = target_path[target_name]
            
            with open(target, 'rb') as fp:
                target_ds = pickle.load(fp)
                print('导入原始数据成功~~~')
                assert target_ds.metadata['dim'] == 1, 'target 序列的维度都应该为1'
                target_data = target_ds

                prophet_predictor = RForecastPredictor(freq=freq, prediction_length=pred
                                    ,method_name='arima' ,trunc_length=past)
                generators = prophet_predictor.predict(target_data.train)
                forecast_samples = list(generators)
                sorted_samples = np.concatenate([np.expand_dims(sample._sorted_samples, 0) for sample in forecast_samples], axis=0)
                sorted_samples = np.expand_dims(sorted_samples, axis=0)
                sample_forecasts.append(sorted_samples)

        sample_forecasts = np.concatenate(sample_forecasts, axis=0)
        print('把预测结果保存在-->', forecast_result_saved_path)
        with open(forecast_result_saved_path , 'wb') as fp:
            pickle.dump(sample_forecasts , fp)
