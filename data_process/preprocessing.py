# -*- coding: UTF-8 -*-
# author : joelonglin
import numpy as np
import pandas as pd
from typing import List, NamedTuple, Optional
from gluonts.dataset.common import ListDataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.field_names import FieldName
import pickle
import os
import argparse

parser = argparse.ArgumentParser(description="data")
parser.add_argument('-st' ,'--start' , type=str , help='数据集开始的时间', default='2018-08-02')
parser.add_argument('-d','--dataset', type=str, help='需要重新生成的数据的名称',default='btc')
parser.add_argument('-t','--train_length', type=int, help='数据集训练长度', default=30)
parser.add_argument('-p'  ,'--pred_length' , type = int , help = '需要预测的长度' , default=1)
parser.add_argument('-s'  ,'--slice' , type = str , help = '需要预测的长度' , default='nolap')
parser.add_argument('-n' , '--num_time_steps' , type=int  , help='时间步的数量' , default=637)
parser.add_argument('-f' , '--freq' , type=str  , help='时间间隔' , default='1D')
args = parser.parse_args()

root='data_process/raw_data/'
class DatasetInfo(NamedTuple):
    name:str  # 该数据集的名称
    url:str  #存放该数据集的
    time_col : str  #表明这些 url 里面对应的 表示时刻的名称 如: eth.csv 中的 beijing_time  或者 btc.csv 中的Date
    dim: int  #该数据集存在多少条序列
    aim : List[str]  # 序列的目标特征，包含的数量特征应该与 url， time_col一致
    feat_dynamic_cat: Optional[str] = None  #表明该序列类别的 (seq_length , category)

datasets_info = {
    "btc": DatasetInfo(
        name="btc",
        url=root + 'btc.csv',
        time_col='beijing_time',
        dim=1,
        aim=['close'],
    ),
    "btc_diff": DatasetInfo(
        name="btc_diff",
        url=root + 'btc_diff.csv',
        time_col='beijing_time',
        dim=1,
        aim=['close'],
    ),
    "eth": DatasetInfo(
        name="eth",
        url=root + 'eth.csv',
        time_col='beijing_time',
        dim=1,
        aim=['close'],  # 这样子写只是为了测试预处理程序是否强大
    ),
    "eth_diff": DatasetInfo(
        name="eth_diff",
        url=root + 'eth_diff.csv',
        time_col='beijing_time',
        dim=1,
        aim=['close'],  # 这样子写只是为了测试预处理程序是否强大
    ),
    "gold": DatasetInfo(
        name="gold",
        url=root + 'GOLD.csv',
        time_col='Date',
        dim=2,
        aim=['Open','Close'],
    ),
    "gold_diff": DatasetInfo(
        name="gold_diff",
        url=root + 'GOLD_diff.csv',
        time_col='Date',
        dim=2,
        aim=['Open','Close'],
    ),
    "gold_lbma": DatasetInfo(
        name="gold_lbma",
        url=root + 'gold_lbma.csv',
        time_col='Date',
        dim=2,
        aim=['USD(AM)', 'GBP(AM)'],
    ),
    "gold_lbma_diff": DatasetInfo(
        name="gold_lbma_diff",
        url=root + 'gold_lbma_diff.csv',
        time_col='Date',
        dim=2,
        aim=['USD(AM)', 'GBP(AM)'],
    ),
}
# 切割完之后， 除了目标序列target_slice 之外
# 我去掉了DeepState 里面的 feat_static_cat, 因为真的不用给样本进行编号
# 此方法用于 stride = 1, 完全滑动窗口
def slice_df_overlap(
    dataframe ,window_size
):
    '''
    :param dataframe:  给定需要滑动切片的总数据集
    :param window_size:  窗口的大小
    :return: (1)序列开始的时间 List[start_time] (2) 切片后的序列
    '''
    data = dataframe.values
    timestamp = dataframe.index
    target_slice,target_start = [], []
    _ , dim = data.shape
    # feat_static_cat 特征值，还是按照原序列进行标注
    for i in range(window_size - 1, len(data)):
        a = data[(i - window_size + 1):(i + 1)]
        target_slice.append(a)
        target_start.append(timestamp[i-window_size+1])

    target_slice = np.array(target_slice)
    return target_slice, target_start

# 此方法对序列的切割，完全不存在重叠
# 缺点：如果窗口和长度不存在倍数关系，会使得数据集存在缺失
def slice_df_nolap(
    dataframe,window_size
):
    data = dataframe.values
    timestamp = dataframe.index
    target_slice, target_start= [], []
    _, dim = data.shape
    series = len(data) // window_size
    for j in range(series):
        a = data[j*window_size : (j+1)*window_size]
        target_slice .append(a)
        target_start.append(timestamp[j*window_size])
    target_slice = np.array(target_slice)
    return target_slice , target_start

# TODO: 这里
slice_func = {
    'overlap' : slice_df_overlap,
    'nolap' : slice_df_nolap
}

def load_finance_from_csv(ds_info):
    df = pd.DataFrame()
    path, time_str,aim  = ds_info.url, ds_info.time_col , ds_info.aim
    series_start = pd.Timestamp(ts_input=args.start , freq=args.freq)
    series_end = series_start + (args.num_time_steps-1)*series_start.freq
    url_series = pd.read_csv(path ,sep=',' ,header=0 , parse_dates=[time_str])
    # TODO: 这里要注意不够 general 只在适合 freq='D' 的情况
    if 'D' in args.freq:
        url_series[time_str] = pd.to_datetime(url_series[time_str].dt.date)
    url_series.set_index(time_str ,inplace=True)
    url_series = url_series.loc[series_start:series_end][aim]
    if url_series.shape[0] < args.num_time_steps:
        #添加缺失的时刻(比如周末数据确实这样的)
        index = pd.date_range(start=series_start, periods=args.num_time_steps, freq=series_start.freq)
        pd_empty = pd.DataFrame(index=index)
        url_series = pd_empty.merge(right=url_series,left_index=True , right_index=True, how='left')

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
        target_slice , target_start = func(df_aim, window_size)
        ds_metadata['sample_size'] = len(target_slice)
        ds_metadata['num_step'] = window_size
        ds_metadata['start'] = target_start
        return target_slice,ds_metadata
    else: # 表示该数据集不进行切割
        # feat_static_cat = np.arange(ds_info.dim).astype(np.float32)
        ds_metadata['sample_size'] = 1
        ds_metadata['num_step'] = args.num_time_steps
        ds_metadata['start'] = [pd.Timestamp(datasets_info[dataset_name].start_date
                                             ,freq = datasets_info[dataset_name].freq)
                                for _ in range(datasets_info[dataset_name].dim)]
        return np.expand_dims(df_aim.values, 0) , ds_metadata




def createGluontsDataset(data_name):
    ds_info = datasets_info[data_name]
    # 获取所有目标序列，元信息
    #(samples_size , seq_len , 1)
    target_slice, ds_metadata = create_dataset(data_name)
    # print('feat_static_cat : ' , feat_static_cat , '  type:' ,type(feat_static_cat))
    train_ds = ListDataset([{FieldName.TARGET: target,
                             FieldName.START: start,}
                            for (target, start) in zip(target_slice[:, :-ds_metadata['prediction_length']],
                                                        ds_metadata['start'],
                                                        )],
                           freq=ds_metadata['freq'],
                           one_dim_target=False)

    test_ds = ListDataset([{FieldName.TARGET: target,
                            FieldName.START: start,
                            }
                           for (target, start) in zip(target_slice,
                                                       ds_metadata['start'],
                                                       )],
                          freq=ds_metadata['freq'],
                          one_dim_target=False)

    dataset = TrainDatasets(metadata=ds_metadata, train=train_ds, test=test_ds)
    with open('data_process/processed_data/{}_{}_{}_{}.pkl'.format(
            '%s_start(%s)_freq(%s)'%(data_name, args.start,args.freq), '%s_DsSeries_%d'%(args.slice,dataset.metadata['sample_size']),
            'train_%d'%args.train_length, 'pred_%d'%args.pred_length,
    ), 'wb') as fp:
        pickle.dump(dataset, fp)

    print('当前数据集为: ', data_name , '训练长度为 %d , 预测长度为 %d '%(args.train_length , args.pred_length),
          '(切片之后)每个数据集样本数目为：'  , dataset.metadata['sample_size'])

if __name__ == '__main__':
    if 'lzl_shared_ssm' in os.getcwd():
        if 'data_process' in os.getcwd():
            os.chdir('..')
        print('------当前与处理数据的路径：', os.getcwd() ,'-------')
    else:
        print('处理数据的时候，请先进入finance的主目录')
        exit()

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