# -*- coding: UTF-8 -*-
# author : joelonglin
import numpy as np
import pandas as pd
from typing import List, NamedTuple, Optional , Union
from gluonts.dataset.common import ListDataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.field_names import FieldName
import pickle
import os
import argparse

parser = argparse.ArgumentParser(description="data")
parser.add_argument('-st' ,'--start' , type=str , help='数据集开始的时间', default='2014-01-02')
parser.add_argument('-d','--dataset', type=str, help='需要重新生成的数据的名称',default='Aluminium')
parser.add_argument('-t','--train_length', type=int, help='数据集训练长度', default=264)
parser.add_argument('-p'  ,'--pred_length' , type = int , help = '需要预测的长度' , default=1)
parser.add_argument('-s'  ,'--slice' , type = str , help = '需要预测的长度' , default='overlap')
parser.add_argument('-n' , '--num_time_steps' , type=int  , help='时间步的数量' , default=1303)
parser.add_argument('-f' , '--freq' , type=str  , help='时间间隔' , default='1B')
args = parser.parse_args()

raw_root='datasets/compeition_sigir2020/Train/Train_data/'
val_root = 'datasets/compeition_sigir2020/Validation/Validation_data/'
processed_root='data_process/processed_data/'
if not os.path.exists(processed_root):
    os.makedirs(processed_root)




class DatasetInfo(NamedTuple):
    name:str  # 该数据集的名称
    url:Union[str,List[str]]  #存放该数据集的
    time_col : str  #表明这些 url 里面对应的 表示时刻的名称 如: eth.csv 中的 beijing_time  或者 btc.csv 中的Date
    dim: int  #该数据集存在多少条序列
    aim : List[str]  # 序列的目标特征，包含的数量特征应该与 url， time_col一致
    index_col : Optional[int] = None #是否存在多余的一列 index
    feat_dynamic_cat: Optional[str] = None  #表明该序列类别的 (seq_length , category)

datasets_info = {
    "LMEAluminium": DatasetInfo(
        name="Aluminium",
        url=[raw_root + 'LMEAluminium3M_train.csv' , val_root+'LMEAluminium3M_validation.csv'],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMECopper": DatasetInfo(
        name="Copper",
        url= [raw_root + 'LMECopper3M_train.csv', val_root+'LMECopper3M_validation.csv'],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMELead": DatasetInfo(
        name="Lead",
        url=[raw_root + 'LMELead3M_train.csv', val_root+'LMELead3M_validation.csv'],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMENickel": DatasetInfo(
        name="Nickel",
        url=[raw_root + 'LMENickel3M_train.csv', val_root +'LMENickel3M_validation.csv'],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMETin": DatasetInfo(
        name="Tin",
        url=[raw_root + 'LMETin3M_train.csv' , val_root+'LMETin3M_validation.csv'],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMEZinc": DatasetInfo(
        name="Zinc",
        url=[raw_root + 'LMEZinc3M_train.csv' , val_root+'LMEZinc3M_validation.csv'],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
}
# 切割完之后， 除了目标序列target_slice 之外
# 我去掉了DeepState 里面的 feat_static_cat, 因为真的不用给样本进行编号
# 此方法用于 stride = 1, 完全滑动窗口
def slice_df_overlap(
    dataframe ,window_size,past_size
):
    '''
    :param dataframe:  给定需要滑动切片的总数据集
    :param window_size:  窗口的大小
    :param past_size:  训练的窗口大小
    :return: (1)序列开始的时间 List[start_time] (2) 切片后的序列
    '''
    data = dataframe.values
    timestamp = dataframe.index
    target_slice,target_start ,target_forecast_start = [], [] , []
    # feat_static_cat 特征值，还是按照原序列进行标注
    for i in range(window_size - 1, len(data)):
        a = data[(i - window_size + 1):(i + 1)]
        target_slice.append(a)
        start = timestamp[i-window_size+1]
        target_start.append(start)
        forecast_start = start + past_size*start.freq
        target_forecast_start.append(forecast_start)

    target_slice = np.array(target_slice)
    return target_slice, target_start,target_forecast_start

# 此方法对序列的切割，完全不存在重叠
# 缺点：如果窗口和长度不存在倍数关系，会使得数据集存在缺失
def slice_df_nolap(
    dataframe,window_size,past_size
):
    data = dataframe.values
    timestamp = dataframe.index
    target_slice, target_start , target_forecast_start= [], [], []
    series = len(data) // window_size
    for j in range(series):
        a = data[j*window_size : (j+1)*window_size]
        target_slice.append(a)
        start = timestamp[j*window_size]
        target_start.append(start)
        forecast_start = start + past_size*start.freq
        target_forecast_start.append(forecast_start)
    target_slice = np.array(target_slice)
    return target_slice , target_start , target_forecast_start

# TODO: 这里
slice_func = {
    'overlap' : slice_df_overlap,
    'nolap' : slice_df_nolap
}

def load_finance_from_csv(ds_info : DatasetInfo):
    df = pd.DataFrame()
    path, time_str,aim  = ds_info.url, ds_info.time_col , ds_info.aim
    series_start = pd.Timestamp(ts_input=args.start , freq=args.freq)
    series_end = series_start + (args.num_time_steps-1)*series_start.freq
    # 单一数据集
    if isinstance(path , str):
        url_series = pd.read_csv(path ,sep=',' ,header=0 , parse_dates=[time_str])
    # 多个数据集 ，进行 时间轴 上的拼接
    elif isinstance(path , List):
        url_series = pd.DataFrame();
        for file in path:
            file_series = pd.read_csv(file ,sep=',' ,header=0 , index_col=ds_info.index_col, parse_dates=[time_str])
            url_series = pd.concat([url_series , file_series] , axis = 0)
        

    # TODO: 这里要注意不够 general 只在适合 freq='D' 或者 freq = 'B' 的情况
    if 'D' in args.freq or 'B' in args.freq:
        url_series[time_str] = pd.to_datetime(url_series[time_str].dt.date)
    url_series.set_index(time_str ,inplace=True)
    url_series = url_series.loc[series_start:series_end][aim]
    if url_series.shape[0] < args.num_time_steps:
        #添加缺失的时刻(比如周末数据缺失这样的)
        index = pd.date_range(start=series_start, periods=args.num_time_steps, freq=series_start.freq)
        pd_empty = pd.DataFrame(index=index)
        url_series = pd_empty.merge(right=url_series,left_index=True , right_index=True, how='left')
        url_series = url_series.fillna(axis=0, method='ffill')
        
    # TODO: 这里因为当第二个 pd.Series要插入到 df 中出现错误时，切记要从初始数据集的脏数据出发
    # 使用 url_series[col].index.duplicated()  或者  df.index.duplicated() 查看是否存在index重复的问题
    for col in url_series.columns:
        df["{}_{}".format(ds_info.name,col)] = url_series[col]
    return df

def create_dataset(dataset_name):
    ds_info = datasets_info[dataset_name]
    df_aim = load_finance_from_csv(ds_info) #(seq , features)
    ds_metadata = {'prediction_length': args.pred_length,
                   'dim': ds_info.dim,
                   'freq':args.freq,
    }
    func = slice_func.get(args.slice)
    if func != None: #表示存在窗口切割函数
        window_size = args.pred_length + args.train_length
        target_slice , target_start , target_forecast_start = func(df_aim, window_size, args.train_length)
        ds_metadata['sample_size'] = len(target_slice)
        ds_metadata['num_step'] = window_size
        ds_metadata['start'] = target_start
        ds_metadata['forecast_start'] = target_forecast_start
        return target_slice,ds_metadata
    else: # 表示该数据集不进行切割
        # feat_static_cat = np.arange(ds_info.dim).astype(np.float32)
        ds_metadata['sample_size'] = 1
        ds_metadata['num_step'] = args.num_time_steps
        time_start = pd.Timestamp(args.start,freq = args.freq)
        ds_metadata['start'] = [ time_start
                                for _ in range(datasets_info[dataset_name].dim)]
        ds_metadata['forecast_start'] =[time_start + (args.num_time_steps-args.pred_length)*time_start.freq
                                for _ in range(datasets_info[dataset_name].dim)]
        return np.expand_dims(df_aim.values, 0) , ds_metadata




def createGluontsDataset(data_name):
    # 获取所有目标序列，元信息
    #(samples_size , seq_len , 1)
    target_slice, ds_metadata = create_dataset(data_name)
    
    train_ds = ListDataset([{FieldName.TARGET: target,
                             FieldName.START: start,
                             FieldName.FORECAST_START: forecast_start}
                            for (target, start, forecast_start) in zip(target_slice[:, :-ds_metadata['prediction_length']],
                                                        ds_metadata['start'], ds_metadata['forecast_start']
                                                        )],
                           freq=ds_metadata['freq'],
                           one_dim_target=False)

    test_ds = ListDataset([{FieldName.TARGET: target,
                            FieldName.START: start,
                            FieldName.FORECAST_START: forecast_start}
                           for (target, start, forecast_start) in zip(target_slice,
                                                       ds_metadata['start'],ds_metadata['forecast_start']
                                                       )],
                          freq=ds_metadata['freq'],
                          one_dim_target=False)

    dataset = TrainDatasets(metadata=ds_metadata, train=train_ds, test=test_ds)
    
    with open(processed_root+'{}_{}_{}_{}.pkl'.format(
            '%s_start(%s)_freq(%s)'%(data_name, args.start,args.freq), 'steps(%d)_slice(%s)_DsSeries(%d)'%(args.num_time_steps , args.slice, dataset.metadata['sample_size']),
            'train(%d)'%args.train_length, 'pred(%d)'%args.pred_length,
    ), 'wb') as fp:
        pickle.dump(dataset, fp)

    print('当前数据集为: ', data_name , '训练长度为 %d , 预测长度为 %d '%(args.train_length , args.pred_length),
          '(切片之后)每个数据集样本数目为：'  , dataset.metadata['sample_size'])

if __name__ == '__main__':
    finance_data_name = args.dataset
    createGluontsDataset(finance_data_name)




# pandas 设置 date_range的API
# time_index = pd.date_range(
#     start=args.start,
#     freq=args.freq,
#     periods=args.num_time_steps,
# )
# df = df.set_index(time_index)


# 当 series_url 是一个 List的情况下
# def load_finance_from_csv(ds_info):
#     df = pd.DataFrame()
#     for url_no in range(len(ds_info.url)):
#         path, time_str  = ds_info.url[url_no], ds_info.time_col[url_no]
#         if isinstance(ds_info.aim[url_no] , list): #当前数据集 有多个目标序列，或 传入的值为 List
#             aim = ds_info.aim[url_no]
#         elif isinstance(ds_info.aim[url_no] , str): #当前数据集 只有一个目标序列，传入的值是str
#             aim = [ds_info.aim[url_no]]
#         series_start = pd.Timestamp(ts_input=args.start , freq=args.freq)
#         series_end = series_start + (args.num_time_steps-1)*series_start.freq
#         data_name = path.split('/')[-1].split('.')[0]
#         url_series = pd.read_csv(path ,sep=',' ,header=0 , parse_dates=[time_str])
#         if 'D' in args.freq:
#             url_series[time_str] = pd.to_datetime(url_series[time_str].dt.date)
#         url_series.set_index(time_str ,inplace=True)
#         url_series = url_series.loc[series_start:series_end][aim]
#         if url_series.shape[0] < args.num_time_steps:
#             #添加缺失的时刻(比如周末数据确实这样的)
#             index = pd.date_range(start=series_start, periods=args.num_time_steps, freq=series_start.freq)
#             pd_empty = pd.DataFrame(index=index)
#             url_series = pd_empty.merge(right=url_series,left_index=True , right_index=True, how='left')
#
#         # 使用 url_series[col].index.duplicated()  或者  df.index.duplicated() 查看是否存在index重复的问题
#         for col in url_series.columns:
#             df[f"{data_name}_{col}"] = url_series[col]
#     return df