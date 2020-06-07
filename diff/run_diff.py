# -*- coding: UTF-8 -*-
# author : joelonglin

# 直接用差分作为结果
# 使用 T 和 T-pred 的涨跌直接作为 T + pred的涨跌结果
import pandas as pd
import numpy as np
import os
import datetime


if __name__ == '__main__':
    trainpath = 'datasets/compeition_sigir2020/Train/Train_data'
    valpath = 'datasets/compeition_sigir2020/Validation/Validation_data'

    trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']
    valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']
    trainfiles_oi = ['LMEAluminium_OI_train.csv','LMECopper_OI_train.csv','LMELead_OI_train.csv','LMENickel_OI_train.csv','LMETin_OI_train.csv','LMEZinc_OI_train.csv']

    for ind in range(len(trainfiles_3m)):
        data_train = pd.read_csv(os.path.join(trainpath,trainfiles_3m[ind]), delimiter=',',index_col=0,usecols=(1,5))
        data_val = pd.read_csv(os.path.join(valpath,valfiles_3m[ind]),delimiter=',',index_col=0,usecols=(1,5))
        data = pd.concat([data_train , data_val] , axis = 0)
        val_start_index = data_train.shape[1] # the start of validation
        for i in [1, 20, 60]:
            result = data.diff(i)
            result[result <= 0] = 0
            result[result > 0] = 1 
            result = result[len(data_train):]

            ds_name = trainfiles_oi[i].split('_')[0]
            # for x in result:
            #     print('{}-validation-{}d-{}'.format(ds_name,i,x.index))
            # exit()
            result['id'] = result.index.apply(lambda x: '{}-validation-{}d-{}'.format(ds_name,i,x) , axis=0)
            
            result
            pass
            # The train_data doesn't split training set and test set, so we do it manually by pass a string parameter.
            
            # torch.save(mlp,'E:/BaiduNetdiskDownload/compeition_sigir2020/%sday_%s_model'%(i,trainfiles_3m[ind].split('_')[0].strip('3M')))
            

            # break
        break