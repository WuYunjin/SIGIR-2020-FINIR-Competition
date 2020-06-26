import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import GridSearchCV
from sklearn import tree
import xgboost as xgb
import logging

from talib.abstract import *
import argparse

parser = argparse.ArgumentParser(description="xgboost")
parser.add_argument('-t' ,'--train_data_len' , type=int , help='training len', default=500)
parser.add_argument('-v','--valdata_len', type=int, help='valuation len',default=1)
parser.add_argument('-p'  ,'--prob' , type = float , help = 'xgb prob output threshold' , default=0.5)
parser.add_argument('-d','--max_depth', type=int, help='xgb max_length', default=8)
parser.add_argument('-e'  ,'--eta' , type = float , help = 'xgb eta' , default=0.01)
parser.add_argument('-g'  ,'--gamma' , type = float , help = 'xgb gamma' , default=0.0)
parser.add_argument('-m'  ,'--metal' , type = int , help = 'index of the target metal' , default=0)
parser.add_argument('-n' , '--n_estimators' , type=int  , help='num of the estimate time' , default=50)
args = parser.parse_args()

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

        
# extract feature for  ind-th metal
def feature_extract(traindata_len,ind):
    
        day = 1
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
        all_data['diff_1'] = all_data['close'].diff(1)
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

        all_data = all_data.join(pd.concat([train_label,val_label]))
        data = all_data[-traindata_len-253:] #253 is the length of validation set
        return data


def train_xgb(feature,label,params_xgb):
        
       
        xgboost =  xgb.XGBClassifier(random_state=10 , **params_xgb)
        xgboost.fit(feature,label)

        return xgboost

def val():

        logging.basicConfig(
                filename='stage2/XGB/xgb.log',
                level=logging.INFO
        )
        prediction = pd.DataFrame()
        prediction['id'] = []
        prediction['label'] = []
        result = pd.read_csv('result_leak.csv')
        base = np.mean(result.loc[result['id'].str.contains('test-1d')]['label'].values==0)
        
        
        accuracy = 0
        
        # prob_list =[0.55,0.6,0.55,0.55,0.55,0.55] 
        # valdata_len_list = [15,1,15,25,15,1] 
        # train_data_len_list = [600, 600, 550, 550, 500,500] 

        prob = args.prob
        train_data_len = args.train_data_len
        valdata_len = args.valdata_len
        val_dummy = valdata_len
        
        params_xgb = {
        'max_depth': args.max_depth,
        'gamma':args.gamma,
        'eta':args.eta,
        'objective': 'binary:logistic',
        'base_score' : base, # 初始预测得分，全1或者全0的分数
        'n_estimators': args.n_estimators
        }

        # print('the hyperparameter is(train , val , prob) :  ' , train_data_len_list ,' ', valdata_len_list , ' ' , prob_list , ' ')
        print('the xgboost hyperparameter is :  ' , params_xgb)

        

        for ind in range(args.metal , args.metal+1):

                # train_data_len = train_data_len_list[ind]
                # valdata_len = valdata_len_list[ind]
                # val_dummy = valdata_len
                # prob = prob_list[ind]

                data = feature_extract(train_data_len,ind=ind)

                window_start = train_data_len +253
                window_end = 253
                
                print('The target metal is {}'.format(testfiles_oi[ind].split('_')[0]))
                print('the hyperparameter is(train , val , prob) :  ' , train_data_len ,' ', valdata_len , ' ' , prob , ' ')

                flag = 1
                y_pred_all = np.array([])

                
                prefix = valfiles_oi[ind].split('_')[0]+'-test-1d'
                while(flag):
                        # print('for now window_end is {} valdate_len is {}'.format(window_end , valdata_len))
                        if(window_end <= valdata_len):
                            valdata_len =  window_end 
                            flag = 0
                                
                        train_data = data[-window_start:-window_end] 
                        train_feature = train_data[train_data.columns.difference(['label'])]
                        train_label = train_data['label']
                        xgboost = train_xgb(train_feature,train_label,params_xgb)

                        if(flag):
                                val_data = data[-window_end:-window_end+valdata_len]
                        else:
                                val_data = data[-window_end:]
                        val_feature = val_data[val_data.columns.difference(['label'])]

                        y_pred = xgboost.predict_proba(val_feature)[:,1]
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
                if acc > 0.55:
                    logging.info('The target metal is {}'.format(testfiles_oi[ind].split('_')[0]))
                    logging.info('the hyperparameter is(train , val , prob) : %s   %s   %s  ' %(str(train_data_len),str(val_dummy),str(prob)) )
                    logging.info('the xgboost hyperparameter is :  %s ' %(str(params_xgb)) )
                    logging.info("accuracy: %s "%( str(acc) ))
                accuracy += acc
                print("accuracy: ",acc)

                temp = pd.DataFrame({'id':prefix+'-'+data[-253:].index,'label':y_pred_all})

                prediction = prediction.append(temp)
        # logging.info("Average accuracy:%s"%(str(accuracy/6)))
        # print("Average accuracy:",str(accuracy/6))
        

        return prediction


if __name__ == "__main__":

        prediction = val()
        
        # result = pd.read_csv('result_leak.csv')
        # prefix_20d = 'validation-20d'
        # prefix_60d = 'validation-60d'
        # output = prediction.append(result[result['id'].str.contains(prefix_20d)])
        # output = output.append(result[result['id'].str.contains(prefix_60d)])
        # output['label'] = output['label'].astype(int)
        # output.to_csv('staget2/XGB/xgb_result.csv',index=False)
