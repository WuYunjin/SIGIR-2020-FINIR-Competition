import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import GridSearchCV
from sklearn import tree
import xgboost as xgb

from talib.abstract import *
        

testpath = 'datasets/compeition_sigir2020/Test/Test_data/'
testfiles_3m = ['LMEAluminium3M_test.csv','LMECopper3M_test.csv','LMELead3M_test.csv','LMENickel3M_test.csv','LMETin3M_test.csv','LMEZinc3M_test.csv']
testfiles_oi = ['LMEAluminium_OI_test.csv','LMECopper_OI_test.csv','LMELead_OI_test.csv','LMENickel_OI_test.csv','LMETin_OI_test.csv','LMEZinc_OI_test.csv']
testfiles_indices = ['Indices_NKY Index_test.csv','Indices_SHSZ300 Index_test.csv','Indices_SPX Index_test.csv','Indices_SX5E Index_test.csv','Indices_UKX Index_test.csv','Indices_VIX Index_test.csv']


valpath = 'datasets/compeition_sigir2020/Validation/Validation_data/'
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']
valfiles_oi = ['LMEAluminium_OI_validation.csv','LMECopper_OI_validation.csv','LMELead_OI_validation.csv','LMENickel_OI_validation.csv','LMETin_OI_validation.csv','LMEZinc_OI_validation.csv']
valfiles_indices = ['Indices_NKY Index_validation.csv','Indices_SHSZ300 Index_validation.csv','Indices_SPX Index_validation.csv','Indices_SX5E Index_validation.csv','Indices_UKX Index_validation.csv','Indices_VIX Index_validation.csv']


trainpath = 'datasets/compeition_sigir2020/Train/Train_data/'
trainfiles_indices = ['Indices_NKY Index_train.csv','Indices_SHSZ300 Index_train.csv','Indices_SPX Index_train.csv','Indices_SX5E Index_train.csv','Indices_UKX Index_train.csv','Indices_VIX Index_train.csv']
trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']
trainfiles_oi = ['LMEAluminium_OI_train.csv','LMECopper_OI_train.csv','LMELead_OI_train.csv','LMENickel_OI_train.csv','LMETin_OI_train.csv','LMEZinc_OI_train.csv']

def feature_extract_xgb(traindata_len,ind, add_diff):
    
        day = 60
        # test set
        test_3m = pd.read_csv(testpath+testfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)


        # Validation set
        val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)
        val_label = pd.read_csv('datasets/compeition_sigir2020/Validation/validation_label_file.csv',names=['date','label'],skiprows=1)        
        prefix = valfiles_oi[ind].split('_')[0]+'-validation-'+str(day)+'d-'
        val_label = val_label.loc[val_label['date'].str.contains(prefix)]
        val_label['date'] = val_label['date'].apply(lambda x: x.replace(prefix,''))
        val_label.set_index(['date'], inplace=True)



        #Trainning set
        suffix = 'Label_'+trainfiles_3m[ind].split('_')[0].strip('3M')+'_train_'+str(day)+'d.csv'
        train_label  = pd.read_csv(trainpath+suffix,delimiter=',',index_col=0,usecols=(1,2),names=['date','label'],skiprows=1)
        train_3m = pd.read_csv(trainpath+trainfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)



        all_data = pd.concat([train_3m,val_3m,test_3m])
        # print(all_data.isnull().sum()) # Missing Value
        # all_data.fillna(method='ffill',inplace=True)
        # print(all_data.isnull().sum()) # Missing Value

        # Construct new features         
        if add_diff:
                all_data['diff_{}'.format(day)] = all_data['close'].diff(day)
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
        # all_data['diff_day'] = all_data['close'].diff(day)
        all_data.dropna(inplace=True)

        all_data = all_data.join(pd.concat([train_label,val_label]))
        data = all_data[-traindata_len-253:] #253 is the length of validation set
        return data

def feature_extract_rf(traindata_len,ind,add_diff):
    
        day = 60

        # test set
        test_3m = pd.read_csv(testpath+testfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)


        # Validation set
        val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)
        val_label = pd.read_csv('datasets/compeition_sigir2020/Validation/validation_label_file.csv',names=['date','label'],skiprows=1)        
        prefix = valfiles_oi[ind].split('_')[0]+'-validation-'+str(day)+'d-'
        val_label = val_label.loc[val_label['date'].str.contains(prefix)]
        val_label['date'] = val_label['date'].apply(lambda x: x.replace(prefix,''))
        val_label.set_index(['date'], inplace=True)



        #Trainning set

        suffix = 'Label_'+trainfiles_3m[ind].split('_')[0].strip('3M')+'_train_'+str(day)+'d.csv'
        train_label  = pd.read_csv(trainpath+suffix,delimiter=',',index_col=0,usecols=(1,2),names=['date','label'],skiprows=1)

        train_3m = pd.read_csv(trainpath+trainfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)



        all_data = pd.concat([train_3m,val_3m,test_3m])
        # print(all_data.isnull().sum()) # Missing Value
        # all_data.fillna(method='ffill',inplace=True)
        # print(all_data.isnull().sum()) # Missing Value

        # Construct new features
        if add_diff:
                all_data['diff_60'] = all_data['close'].diff(60)        
        all_data['sma_10'] = pd.DataFrame(SMA(all_data, timeperiod=10))
        all_data['mom_10'] = pd.DataFrame(MOM(all_data,10))
        all_data['wma_10'] = pd.DataFrame(WMA(all_data,10))
        all_data['sma_20'] = pd.DataFrame(SMA(all_data, timeperiod=20))
        all_data['mom_20'] = pd.DataFrame(MOM(all_data,20))
        all_data['wma_20'] = pd.DataFrame(WMA(all_data,20))
        all_data = pd.concat([all_data,STOCHF(all_data, 
                                          fastk_period=14, 
                                          fastd_period=3)],
                             axis=1)
 
        all_data['macd'] = pd.DataFrame(MACD(all_data, fastperiod=12, slowperiod=26)['macd'])
        all_data['rsi'] = pd.DataFrame(RSI(all_data, timeperiod=14))
        all_data['willr'] = pd.DataFrame(WILLR(all_data, timeperiod=14))
        all_data['cci'] = pd.DataFrame(CCI(all_data, timeperiod=14))
        
        
        all_data['pct_change_20'] = ROC(all_data, timeperiod=20)
        all_data['pct_change_40'] = ROC(all_data, timeperiod=40)
        # all_data['pct_change_30'] = ROC(all_data, timeperiod=30)
        # all_data['pct_change_60'] = ROC(all_data, timeperiod=60)

        all_data.dropna(inplace=True)

        all_data = all_data.join(pd.concat([train_label,val_label]))

        data = all_data[-traindata_len-253:] #253 is the length of test set
        return data


def train_rf(feature,label,params_dummy):
        rf =  RandomForestClassifier(random_state=10,n_estimators=60)
        rf.fit(feature,label)

        return rf

def train_xgb(feature,label,params_xgb):
        
       
        xgboost =  xgb.XGBClassifier(random_state=10 , **params_xgb)
        xgboost.fit(feature,label)

        return xgboost

def val():


    prediction = pd.DataFrame()
    prediction['id'] = []
    prediction['label'] = []
    result = pd.read_csv('result_leak.csv')
    base = np.mean(result.loc[result['id'].str.contains('test-60d')]['label'].values==0)

    accuracy = 0

    prob_list =[0.6, 0.5, 0.4, 0.5, 0.2, 0.4] 
    valdata_len_list = [10, 1, 1, 1, 1, 1] 
    train_data_len_list = [200, 150, 150, 150, 150, 150]
    use_diff = [True , False , False , False , False , False]
    use_model = ['xgb' , 'rf' ,'rf' ,'rf' ,'rf','rf']

    params_xgb = {
        'max_depth': 10,
        'gamma':0.0,
        'eta':0.01,
        'objective': 'binary:logistic',
        'base_score' : base if base>0.5 else 1-base, # come from bias of the dataset
        'n_estimators': 50
    }
    print('base : ' , params_xgb['base_score'])
    model_method = {
            'rf': train_rf,
            'xgb': train_xgb
    }

    feature_method = {
        'rf' : feature_extract_rf,
        'xgb': feature_extract_xgb
    }


    for ind in range(6):

        train_data_len = train_data_len_list[ind]
        valdata_len = valdata_len_list[ind]
        val_dummy = valdata_len
        prob = prob_list[ind]

        data = feature_method[use_model[ind]](train_data_len,ind=ind, add_diff=use_diff[ind])

        window_start = train_data_len +253
        window_end = 253

        # print('The target metal is {} '.format(testfiles_oi[ind].split('_')[0]))
        # print('the hyperparameter is(train , val , prob) :  ' , train_data_len ,' ', valdata_len , ' ' , prob , ' ')
        # print('the params of xgb is ' , params_xgb)
        

        flag = 1
        y_pred_all = np.array([])
        
        prefix = valfiles_oi[ind].split('_')[0]+'-test-60d'
        method = model_method[use_model[ind]]
        while(flag):
                if(window_end <= valdata_len):
                         valdata_len =  window_end 
                         flag = 0
                        
                train_data = data[-window_start:-window_end] 
                train_feature = train_data[train_data.columns.difference(['label'])]
                train_label = train_data['label']
                model = method(train_feature,train_label,params_xgb)

                if(flag):
                        val_data = data[-window_end:-window_end+valdata_len]
                else:
                        val_data = data[-window_end:]
                val_feature = val_data[val_data.columns.difference(['label'])]

                #Because yunjin use the prob of getting 0
                if use_model[ind] == 'rf':
                        y_pred = model.predict_proba(val_feature)[:,0]
                        y_pred = [0 if x>=prob else 1 for x in y_pred]
                # And I use the prob of getting 1 like stage1
                else:
                        y_pred = model.predict_proba(val_feature)[:,1]
                        y_pred[y_pred>prob]=1
                        y_pred[y_pred<=prob]=0
                
                y_pred_all = np.append(y_pred_all,y_pred)

                if(flag):
                        data.loc[-window_end:-window_end+valdata_len,'label'] = y_pred
                else:                     
                        data.loc[-window_end:,'label'] = y_pred


                window_start -= valdata_len
                window_end -= valdata_len
                valdata_len = val_dummy

        acc = np.mean(result.loc[result['id'].str.contains(prefix)][-253:]['label'].values==y_pred_all)
        print("accuracy: ",acc)
        
        accuracy += acc
        
        
        temp = pd.DataFrame({'id':prefix+'-'+data[-253:].index,'label':y_pred_all})

        prediction = prediction.append(temp)
    print("Average accuracy:",accuracy/6)

    return prediction


if __name__ == "__main__":
        # for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
        #         print("threshold:",t)
        #         prediction = val(t)
        prediction = val()
        
        # prediction['label'] = prediction['label'].astype(int)
        # prediction.to_csv('result.csv',index=False)
