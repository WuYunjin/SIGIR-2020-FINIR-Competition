import pandas as pd
import numpy as np

valpath = 'datasets/compeition_sigir2020/Test/Test_data/'
prediction = pd.DataFrame()
prediction['id'] = []
prediction['label'] = []

valfiles_oi = ['LMEAluminium_OI_test.csv','LMECopper_OI_test.csv','LMELead_OI_test.csv','LMENickel_OI_test.csv','LMETin_OI_test.csv','LMEZinc_OI_test.csv']
valfiles_3m = ['LMEAluminium3M_test.csv','LMECopper3M_test.csv','LMELead3M_test.csv','LMENickel3M_test.csv','LMETin3M_test.csv','LMEZinc3M_test.csv']


for ind in range(len(valfiles_oi)):
    

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
    prefix = valfiles_oi[ind].split('_')[0]+'-test-1d-'
    print(prefix)
    val_index = prefix + val_oi.index
    temp = pd.DataFrame({'id':val_index,'label':np.array(pred)})
    prediction = prediction.append(temp)

    price_20d = price.diff(20)
    price_20d[price_20d>0] = 1
    price_20d[price_20d<=0] = 0
    pred = np.append(price_20d.values[20:],np.array([price_20d.values[-1]]*20))
    prefix = valfiles_oi[ind].split('_')[0]+'-test-20d-'
    val_index = prefix + val_oi.index
    temp = pd.DataFrame({'id':val_index,'label':np.array(pred)})
    prediction = prediction.append(temp)

    price_60d = price.diff(60)
    price_60d[price_60d>0] = 1
    price_60d[price_60d<=0] = 0
    pred = np.append(price_60d.values[60:],np.array([price_60d.values[-1]]*60))
    prefix = valfiles_oi[ind].split('_')[0]+'-test-60d-'
    val_index = prefix + val_oi.index
    temp = pd.DataFrame({'id':val_index,'label':np.array(pred)})
    prediction = prediction.append(temp)

prediction['label'] = prediction['label'].astype(int)
prediction.to_csv('result_leak.csv',index=False)
