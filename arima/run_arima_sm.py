# -*- coding: UTF-8 -*-
# author : joelonglin
import pandas as pd
import os
import sys
#将当前工作路径添加进去
sys.path.insert(0, '.')
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer



trainpath = 'datasets/compeition_sigir2020/Train/Train_data/'
valpath = 'datasets/compeition_sigir2020/Validation/Validation_data/'
valfiles_oi = ['LMEAluminium_OI_validation.csv','LMECopper_OI_validation.csv','LMELead_OI_validation.csv','LMENickel_OI_validation.csv','LMETin_OI_validation.csv','LMEZinc_OI_validation.csv']
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']
trainfiles_oi = ['LMEAluminium_OI_train.csv','LMECopper_OI_train.csv','LMELead_OI_train.csv','LMENickel_OI_train.csv','LMETin_OI_train.csv','LMEZinc_OI_train.csv']
trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']

# 是否将原始数据进行差分
use_diff = False
prediction = pd.DataFrame()
timestep = 1303 # to the day of 2018-12-31
past = 264 
slice_style = 'overlap'
ds_name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'

prediction = pd.DataFrame()
prediction['id'] = []
prediction['label'] = []

if __name__ == '__main__':
    for ind in range(len(valfiles_oi)):
        for pred in [1,20,60]:
     
            train_3m = pd.read_csv(trainpath+trainfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,5))
            val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,5))
            data = train_3m.append(val_3m)
            feature = 'Close.Price'
            price = data[feature]
            if use_diff:
                #将目标序列从原始序列变成 差分序列
                price = price.diff(1)[train_3m.shape[0]-past-pred+1:]
            else:
                price = price[train_3m.shape[0]-past-pred+1:]
            
            prefix = valfiles_oi[ind].split('_')[0]+'-validation-{}d-'.format(pred)
            #滑动窗口
            for i in range(past+pred-1, len(data)):
                print('===========当前训练的是{}数据集，目标节点是{}=================='.format(
                    valfiles_oi[ind].split('_')[0] , val_3m.index[(i - (past+pred) + 1)]))
                sample = price[(i - (past+pred) + 1):(i + 1)]
                train, test = train_test_split(sample, train_size=past)
                pipeline = Pipeline([
                    # ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),  # lmbda2 avoids negative values
                    ('arima', pm.AutoARIMA(seasonal=True, m=1, 
                                        suppress_warnings=True,
                                        trace=True,error_action="ignore"))
                ])

                pipeline.fit(train )
                pred_result = pipeline.predict(pred)
                print('pred_result is : ' , pred_result)
                print('====================一次训练结束=============================\n\n\n')
                
                val_index = prefix + val_3m.index[(i - (past+pred) + 1)]
                if use_diff:
                    val_label = 1 if pred_result[-1] > 0 else 0
                else:
                    val_label = 1 if pred_result[-1] - train[-1] > 0 else 0
                prediction = prediction.append({'id':val_index , 'label':val_label} , ignore_index=True)

                
            
    prediction['label'] = prediction['label'].astype(int)
    prediction.to_csv('output/arima_train_{}_diff_{}_noBoxCox.csv'.format(past,use_diff),index=False)
            


    