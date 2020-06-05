# -*- coding: UTF-8 -*-
# author : joelonglin

# 直接用差分作为结果
# 使用 T 和 T-pred 的涨跌直接作为 T + pred的涨跌结果
import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    trainpath = 'datasets/compeition_sigir2020/Train/Train_data'

    trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']

    train_output = 'output/train_loss/mlp' #命名以 model 进行命名
    if not os.path.exists(train_output):
        os.makedirs(train_output)

    for ind in range(len(trainfiles_3m)):
        

        for i in [1, 20, 60]:
            pass
            # The train_data doesn't split training set and test set, so we do it manually by pass a string parameter.
            
            # torch.save(mlp,'E:/BaiduNetdiskDownload/compeition_sigir2020/%sday_%s_model'%(i,trainfiles_3m[ind].split('_')[0].strip('3M')))
            

            # break
        break