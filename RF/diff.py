import pandas as pd
import numpy as np


def pseudo_label():
    trainpath = 'datasets/compeition_sigir2020/Train/Train_data/'

    files_oi = ['LMEAluminium_OI_train.csv','LMECopper_OI_train.csv','LMELead_OI_train.csv','LMENickel_OI_train.csv','LMETin_OI_train.csv','LMEZinc_OI_train.csv']
    files_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']

    label_1d = ['Label_LMEAluminium_train_1d.csv','Label_LMECopper_train_1d.csv','Label_LMELead_train_1d.csv','Label_LMENickel_train_1d.csv','Label_LMETin_train_1d.csv','Label_LMEZinc_train_1d.csv']
    label_20d = ['Label_LMEAluminium_train_20d.csv','Label_LMECopper_train_20d.csv','Label_LMELead_train_20d.csv','Label_LMENickel_train_20d.csv','Label_LMETin_train_20d.csv','Label_LMEZinc_train_20d.csv']
    label_60d = ['Label_LMEAluminium_train_60d.csv','Label_LMECopper_train_60d.csv','Label_LMELead_train_60d.csv','Label_LMENickel_train_60d.csv','Label_LMETin_train_60d.csv','Label_LMEZinc_train_60d.csv']


    valpath = 'datasets/compeition_sigir2020/Validation/Validation_data/'
    prediction = pd.DataFrame()
    prediction['id'] = []
    prediction['label'] = []

    valfiles_oi = ['LMEAluminium_OI_validation.csv','LMECopper_OI_validation.csv','LMELead_OI_validation.csv','LMENickel_OI_validation.csv','LMETin_OI_validation.csv','LMEZinc_OI_validation.csv']
    valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']


    for ind in range(len(files_oi)):


        val_oi = pd.read_csv(valpath+valfiles_oi[ind],delimiter=',',index_col=0,usecols=(1,2))
        val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6))
        val_data = val_oi.join(val_3m) # 6-dimension

        data  =  val_data.dropna(axis=0, how='any')
        feature = 'Close.Price'
        price = data[feature]

        price_1d = price.diff(1)
        price_1d[price_1d>0] = 1
        price_1d[price_1d<=0] = 0
        pred = np.append(price_1d.values[1:],0)
        prefix = valfiles_oi[ind].split('_')[0]+'-validation-1d-'
        print(prefix)
        val_index = prefix + val_oi.index
        temp = pd.DataFrame({'id':val_index,'label':np.array(pred)})
        prediction = prediction.append(temp)

        price_20d = price.diff(20)
        price_20d[price_20d>0] = 1
        price_20d[price_20d<=0] = 0
        pred = np.append(price_20d.values[20:],np.array([1]*20))
        prefix = valfiles_oi[ind].split('_')[0]+'-validation-20d-'
        val_index = prefix + val_oi.index
        temp = pd.DataFrame({'id':val_index,'label':np.array(pred)})
        prediction = prediction.append(temp)

        price_60d = price.diff(60)
        price_60d[price_60d>0] = 1
        price_60d[price_60d<=0] = 0
        pred = np.append(price_60d.values[60:],np.array([1]*60))
        prefix = valfiles_oi[ind].split('_')[0]+'-validation-60d-'
        val_index = prefix + val_oi.index
        temp = pd.DataFrame({'id':val_index,'label':np.array(pred)})
        prediction = prediction.append(temp)


    prediction['label'] = prediction['label'].astype(int)
    prediction.to_csv('leak_result.csv',index=False)


if __name__ == "__main__":
    pseudo_label()
    