import pandas as pd
import numpy as np

trainpath = 'datasets/compeition_sigir2020/Train/Train_data/'

trainfiles_oi = ['LMEAluminium_OI_train.csv','LMECopper_OI_train.csv','LMELead_OI_train.csv','LMENickel_OI_train.csv','LMETin_OI_train.csv','LMEZinc_OI_train.csv']
trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']

valpath = 'datasets/compeition_sigir2020/Validation/Validation_data/'
prediction = pd.DataFrame()
prediction['id'] = []
prediction['label'] = []

valfiles_oi = ['LMEAluminium_OI_validation.csv','LMECopper_OI_validation.csv','LMELead_OI_validation.csv','LMENickel_OI_validation.csv','LMETin_OI_validation.csv','LMEZinc_OI_validation.csv']
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']


for ind in range(len(valfiles_oi)):
     
    train_3m = pd.read_csv(trainpath+trainfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,5))
    val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,5))
    data = train_3m.append(val_3m)
    feature = 'Close.Price'
    price = data[feature]

    price_1d = price.diff(1)[-len(val_3m):]
    price_1d[price_1d>0] = 1
    price_1d[price_1d<=0] = 0
    # pred = np.append(price_1d.values[1:],0)
    prefix = valfiles_oi[ind].split('_')[0]+'-validation-1d-'
    print(prefix)
    val_index = prefix + val_3m.index
    temp = pd.DataFrame({'id':val_index,'label':price_1d.values})
    prediction = prediction.append(temp)

    price_20d = price.diff(20)[-len(val_3m):]
    price_20d[price_20d>0] = 1
    price_20d[price_20d<=0] = 0
    # pred = np.append(price_1d.values[1:],0)
    prefix = valfiles_oi[ind].split('_')[0]+'-validation-20d-'
    print(prefix)
    val_index = prefix + val_3m.index
    temp = pd.DataFrame({'id':val_index,'label':price_20d.values})
    prediction = prediction.append(temp)

    price_60d = price.diff(60)[-len(val_3m):]
    price_60d[price_60d>0] = 1
    price_60d[price_60d<=0] = 0
    # pred = np.append(price_1d.values[1:],0)
    prefix = valfiles_oi[ind].split('_')[0]+'-validation-60d-'
    print(prefix)
    val_index = prefix + val_3m.index
    temp = pd.DataFrame({'id':val_index,'label':price_60d.values})
    prediction = prediction.append(temp)
    

prediction['label'] = prediction['label'].astype(int)
prediction.to_csv('output/result.csv',index=False)
