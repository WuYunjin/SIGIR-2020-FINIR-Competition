# -*- coding: UTF-8 -*-
# author : joelonglin
import pandas as pd
import os
import sys
#将当前工作路径添加进去
sys.path.insert(0, '.')
from data_process.utils import create_dataset_if_not_exist , add_time_mark_to_file
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer



target_processed_data = 'LMEAluminium,LMECopper,LMELead,LMENickel,LMETin,LMEZinc'
start = '2014-01-02'
freq = '1B'
timestep = 1303 # to the day of 2018-12-31
pred = 1
past = 264 
slice_style = 'overlap'
ds_name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'

result_params = '_'.join([
                                'freq(%s)'%(freq),
                               'past(%d)'%(past)
                                ,'pred(%d)'%(pred)
                             ]
                        )

if __name__ == '__main__':
    # 导入 target 的数据
    if slice_style == 'overlap':
        series = timestep - past - pred + 1
        print('每个数据集的序列数量为 ', series)
    elif slice_style == 'nolap':
        series = timestep // (past + pred)
        print('每个数据集的序列数量为 ', series)
    else:
        series = 1
        print('每个数据集的序列数量为 ', series ,'情景为长单序列')
    result_root_path  = 'evaluate/results/{}_slice({})_past({})_pred({})'.format(target_processed_data.replace(',' ,'_') , slice_style ,past , pred)
    if not os.path.exists(result_root_path):
        os.makedirs(result_root_path)
    forecast_result_saved_path = os.path.join(result_root_path,'prophet(%s)_' % (target_processed_data.replace(',' ,'_')) + result_params + '.pkl')
    forecast_result_saved_path = add_time_mark_to_file(forecast_result_saved_path)

     # 目标序列的数据路径
    target_path = {ds_name: ds_name_prefix.format(
            '%s_start(%s)_freq(%s)'%(ds_name, start,freq), 'steps(%d)_slice(%s)_DsSeries(%d)'%(timestep , slice_style, series),
            'train(%d)'%past, 'pred(%d)'%pred,
    ) for ds_name in target_processed_data.split(',')}

    create_dataset_if_not_exist(
        paths=target_path, start=start, past_length=past
        , pred_length=pred, slice=slice_style
        , timestep=timestep, freq=freq
    )

    if not os.path.exists(forecast_result_saved_path):
        sample_forecasts = []
        
        for ds_name in target_path:
            target_processed_data = target_path[ds_name]
            
            with open(target_processed_data, 'rb') as fp:
                target_ds = pickle.load(fp)
                print('导入原始数据成功~~~')
                assert target_ds.metadata['dim'] == 1, 'target 序列的维度都应该为1'
                
                train_result = []
                val_result = []
                for entry in target_ds.train:
                    pipeline = Pipeline([
                        ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),  # lmbda2 avoids negative values
                        ('arima', pm.AutoARIMA(seasonal=True, m=12,
                                            suppress_warnings=True,
                                            trace=True))
                    ])

                    pipeline.fit(entry['target'].squeeze())
                    result = pipeline.predict(pred)

                    #预测时间大于等于2018-01-02开始算验证集的结果
                    if entry['forecast_start'] >= pd.Timestamp('2018-01-02'):
                        val_result.append(result)
                    else:
                        train_result.append(result)
                    
                    
                    
                    # Serialize your model just like you would in scikit:
                    # with open('arima_past({}).pkl'.format(past), 'wb') as pkl:
                    #     pickle.dump(pipeline, pkl)
                
                        
                        

        sample_forecasts = np.concatenate(sample_forecasts, axis=0)
        print('把预测结果保存在-->', forecast_result_saved_path)
        with open(forecast_result_saved_path , 'wb') as fp:
            pickle.dump(sample_forecasts , fp)


    