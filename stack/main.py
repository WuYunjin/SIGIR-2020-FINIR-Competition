# -*- coding: UTF-8 -*-
# author :  zhaolong
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')
import tensorflow as tf
from talib.abstract import *
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import *
import seaborn as sns
sns.set()
import pandas as pd

import argparse
import time
import model
from datetime import datetime
from datetime import timedelta

from multiprocessing.dummy import Pool as ThreadPool




parser = argparse.ArgumentParser(description="lstm")
parser.add_argument('-nl' ,'--num_layers' , type=str , help='lstm所需要的层数', default=1)
parser.add_argument('-sl','--size_layer', type=str, help='lstm隐藏层的大小',default=128)
parser.add_argument('-t','--timestamp', type=int, help='训练给定的长度', default=5)
parser.add_argument('-ep'  ,'--epoch' , type = int , help = '需要训练的epoch数量' , default=300)
parser.add_argument('-dp'  ,'--dropout_rate' , type = str , help = 'dropout' , default=0.8)
parser.add_argument('-f' , '--future_day' , type=int  , help='需要预测的时间长度' , default=637)
parser.add_argument('-lr' , '--learning_rate' , type=str  , help='学习率' , default='1D')
args = parser.parse_args()


valpath = 'datasets/compeition_sigir2020/Validation/Validation_data/'
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']
valfiles_oi = ['LMEAluminium_OI_validation.csv','LMECopper_OI_validation.csv','LMELead_OI_validation.csv','LMENickel_OI_validation.csv','LMETin_OI_validation.csv','LMEZinc_OI_validation.csv']
valfiles_indices = ['Indices_NKY Index_validation.csv','Indices_SHSZ300 Index_validation.csv','Indices_SPX Index_validation.csv','Indices_SX5E Index_validation.csv','Indices_UKX Index_validation.csv','Indices_VIX Index_validation.csv']


trainpath = 'datasets/compeition_sigir2020/Train/Train_data/'
trainfiles_indices = ['Indices_NKY Index_train.csv','Indices_SHSZ300 Index_train.csv','Indices_SPX Index_train.csv','Indices_SX5E Index_train.csv','Indices_UKX Index_train.csv','Indices_VIX Index_train.csv']
trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']
trainfiles_oi = ['LMEAluminium_OI_train.csv','LMECopper_OI_train.csv','LMELead_OI_train.csv','LMENickel_OI_train.csv','LMETin_OI_train.csv','LMEZinc_OI_train.csv']


def feature_extract(traindata_len,ind,day):
    
        # Validation set
        val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)


        #Trainning set

        suffix = 'Label_'+trainfiles_3m[ind].split('_')[0].strip('3M')+'_train_'+str(day)+'d.csv'
        train_label  = pd.read_csv(trainpath+suffix,delimiter=',',index_col=0,usecols=(1,2),names=['date','label'],skiprows=1)

        train_3m = pd.read_csv(trainpath+trainfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)

        all_data = pd.concat([train_3m,val_3m])

        # Construct new features         
        
        all_data['sma_10'] = pd.DataFrame(SMA(all_data, timeperiod=10))
        all_data['mom_10'] = pd.DataFrame(MOM(all_data,10))
        all_data['wma_10'] = pd.DataFrame(WMA(all_data,10))
        all_data = pd.concat([all_data,STOCHF(all_data, 
                                          fastk_period=14, 
                                          fastd_period=3)],
                             axis=1)
 
        all_data['macd'] = pd.DataFrame(MACD(all_data, fastperiod=12, slowperiod=26)['macd'])
        all_data['rsi'] = pd.DataFrame(RSI(all_data, timeperiod=14))
        all_data['willr'] = pd.DataFrame(WILLR(all_data, timeperiod=14))
        all_data['cci'] = pd.DataFrame(CCI(all_data, timeperiod=14))
        
        all_data['pct_change_20'] = ROC(all_data, timeperiod=20)
        all_data['pct_change_30'] = ROC(all_data, timeperiod=30)
        all_data['pct_change_60'] = ROC(all_data, timeperiod=60)

        all_data.dropna(inplace=True)

        all_data = all_data.join(train_label)
        data = all_data[-traindata_len-253:] #253 is the length of validation set
        
        return data

def train_and_forecast_1d(train_feature , train_close):
    minmax = MinMaxScaler(feature_range=(-1,1)).fit(train_feature.values)
    train_feature_scaled = minmax.transform(train_feature.astype('float32'))
    # from 1 to Train_end-1
    train_feature_scaled = pd.DataFrame(train_feature_scaled)
    # from 1 to Train_end
    target_scaler = MinMaxScaler(feature_range=(0,1))
    target_scaler = target_scaler.fit(train_close.values.reshape(-1,1))
    train_close_scaled = target_scaler.transform(train_close.values.reshape(-1,1)).reshape(-1)

    ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
    bagging = BaggingRegressor(n_estimators=500)
    et = ExtraTreesRegressor(n_estimators=500)
    gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)
    rf = RandomForestRegressor(n_estimators=500)

    
    ada.fit(train_feature_scaled, train_close_scaled[1:])
    bagging.fit(train_feature_scaled, train_close_scaled[1:])
    et.fit(train_feature_scaled,train_close_scaled[1:])
    gb.fit(train_feature_scaled, train_close_scaled[1:])
    rf.fit(train_feature_scaled, train_close_scaled[1:])

    
    ada_pred=ada.predict(train_feature_scaled)  # from 2 : Train_end
    bagging_pred=bagging.predict(train_feature_scaled)
    et_pred=et.predict(train_feature_scaled)
    gb_pred=gb.predict(train_feature_scaled)
    rf_pred=rf.predict(train_feature_scaled)

    ada_actual = np.hstack([train_close_scaled[0],ada_pred]) # from 1 : Train_end
    bagging_actual = np.hstack([train_close_scaled[0],bagging_pred])
    et_actual = np.hstack([train_close_scaled[0],et_pred])
    gb_actual = np.hstack([train_close_scaled[0],gb_pred])
    rf_actual = np.hstack([train_close_scaled[0],rf_pred])
    stack_predict = np.vstack([ada_actual,bagging_actual,et_actual,gb_actual,rf_actual,train_close_scaled]).T #(train_len , models_num+1) from 1 : T

    
    import xgboost as xgb
    params_xgd = {
        'max_depth': 10,
        'objective': 'reg:logistic',
        'learning_rate': 0.1,
        'n_estimators': 1000
        }
    train_Y = train_close_scaled[1:] # from 2 : Train_end
    clf = xgb.XGBRegressor(**params_xgd)
    #use the prediction of sklearn regressor as the input of XGB
    clf.fit(stack_predict[:-1,:],train_Y, eval_set=[(stack_predict[:-1,:],train_Y)], 
            eval_metric='rmse', early_stopping_rounds=20, verbose=False)
    
    xgb_pred = clf.predict(stack_predict)# from 2 : T+1
    xgb_pred = target_scaler.inverse_transform(xgb_pred.reshape(-1,1))

    return xgb_pred[-1]

def process_single_dataset(ind):
    accuracy =0
    traindata_len = 500 # window_size to train
    data = feature_extract(traindata_len,ind=ind , day=1)

    window_start = 1+traindata_len +253
    window_end = 252 # we want to  use -253(2018-01-02) to train
    y_pred_all = np.array([])
    
    result = pd.read_csv('result_93.58.csv')
    prefix = valfiles_oi[ind].split('_')[0]+'-validation-1d'
    print('======== {} prediction is beginning ========'.format(valfiles_oi[ind].split('_')[0]))
    while(window_end >= 0):
            
            
            if window_end == 0:
                negative_window_end = None
            else:
                negative_window_end = -window_end
            train_data = data[-window_start : negative_window_end] #(train_len , all_features)
            train_feature = train_data[train_data.columns.difference(['label'])].iloc[:-1] #(train_len-1 , features)
            train_close = train_data['close'] #(train_len,)

            xgb_result = train_and_forecast_1d(train_feature , train_close)
            print('{} current predict is one day after {}({}) , and it is {}'.format(valfiles_oi[ind].split('_')[0],train_data.index[-1], train_data.iloc[-1]['close'] , xgb_result))
            label = 1 if xgb_result - train_data.iloc[-1]['close'] > 0 else 0

            y_pred_all = np.append(y_pred_all , label)
            
            window_start -= 1
            
            window_end -= 1
    # print('======== {} prediction is end ========\n\n\n\n'.format(valfiles_oi[ind].split('_')[0]))
    # print(y_pred_all)
    acc = np.mean(result.loc[result['id'].str.contains(prefix)][-253:]['label'].values==y_pred_all)
    accuracy += acc
    print("{} accuracy: {} \n\n".format(valfiles_oi[ind].split('_')[0],acc))
    
    temp = pd.DataFrame({'id':prefix+'-'+data[-253:].index,'label':y_pred_all})

    return temp, accuracy;
    # return pd.DataFrame({'id':np.zeros(100) ,'label':np.ones(100)}) , 100
        

def val_1d():
    prediction = pd.DataFrame()
    prediction['id'] = []
    prediction['label'] = []

    accuracy = 0    
    # for ind in range(6):
    pool = ThreadPool(processes=6)
    
    # single_output = pool.apply_async(process_single_dataset,list(range(6)),)
    multi_res = [pool.apply_async(process_single_dataset, (i,)) for i in range(6)]
   

    for res in multi_res:
        res = res.get()
        prediction = prediction.append(res[0])
        single_acc = res[1]
        accuracy += single_acc
       
    # print("Average accuracy:",accuracy/6)

    return prediction

if __name__ == '__main__':
    
    print("Start Now，will take few minutes")
    time_start=time.time()
    prediction = val_1d()
    prediction['label'] = prediction['label'].astype(int)
    prediction.to_csv('output/stack_ensemble_xgb.csv' ,index=False)




























    # import os
    # if not os.path.exists('output/stack'):
    #     os.makedirs('output/stack')

    # plt.xticks(rotation=270,fontsize=8)
    # plt.bar(train_feature.columns, ada.feature_importances_)
    # plt.title('ada boost important feature')
    # plt.savefig('output/stack/ada_importance.png')
    # plt.clf()
    

    # plt.xticks(rotation=270,fontsize=8)
    # plt.bar(train_feature.columns, et.feature_importances_)
    # plt.title('et important feature')
    # plt.savefig('output/stack/et_importance.png')
    # plt.clf()

    # plt.xticks(rotation=270,fontsize=8)
    # plt.bar(train_feature.columns, gb.feature_importances_)
    # plt.title('gradient boost important feature')
    # plt.savefig('output/stack/gb_importance.png')
    # plt.clf()

    # plt.xticks(rotation=270,fontsize=8)
    # plt.bar(train_feature.columns, rf.feature_importances_)
    # plt.title('RF important feature')
    # plt.savefig('output/stack/rf_importance.png')
    # plt.clf()