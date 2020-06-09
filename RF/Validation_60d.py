import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import GridSearchCV
from sklearn import tree


        
valpath = 'datasets/compeition_sigir2020/Validation/Validation_data/'
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']
valfiles_oi = ['LMEAluminium_OI_validation.csv','LMECopper_OI_validation.csv','LMELead_OI_validation.csv','LMENickel_OI_validation.csv','LMETin_OI_validation.csv','LMEZinc_OI_validation.csv']
valfiles_indices = ['Indices_NKY Index_validation.csv','Indices_SHSZ300 Index_validation.csv','Indices_SPX Index_validation.csv','Indices_SX5E Index_validation.csv','Indices_UKX Index_validation.csv','Indices_VIX Index_validation.csv']


trainpath = 'datasets/compeition_sigir2020/Train/Train_data/'
trainfiles_indices = ['Indices_NKY Index_train.csv','Indices_SHSZ300 Index_train.csv','Indices_SPX Index_train.csv','Indices_SX5E Index_train.csv','Indices_UKX Index_train.csv','Indices_VIX Index_train.csv']
trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']
trainfiles_oi = ['LMEAluminium_OI_train.csv','LMECopper_OI_train.csv','LMELead_OI_train.csv','LMENickel_OI_train.csv','LMETin_OI_train.csv','LMEZinc_OI_train.csv']

        

def feature_extract(traindata_len,ind):
    
        day = 60
        # Validation set

        val_oi = pd.read_csv(valpath+valfiles_oi[ind],delimiter=',',index_col=0,usecols=(1,2))
        val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,5,6))#usecols=(1,2,3,4,5,6))

        val_nky = pd.read_csv(valpath+valfiles_indices[0],delimiter=',',index_col=0,usecols=(1,2))
        val_shsz300 = pd.read_csv(valpath+valfiles_indices[1],delimiter=',',index_col=0,usecols=(1,2))        
        val_spx = pd.read_csv(valpath+valfiles_indices[2],delimiter=',',index_col=0,usecols=(1,2))
        val_sx5e = pd.read_csv(valpath+valfiles_indices[3],delimiter=',',index_col=0,usecols=(1,2))
        val_ukx = pd.read_csv(valpath+valfiles_indices[4],delimiter=',',index_col=0,usecols=(1,2))
        val_vix = pd.read_csv(valpath+valfiles_indices[5],delimiter=',',index_col=0,usecols=(1,2))

        val_data = val_oi.join(val_3m).join(val_nky).join(val_shsz300).join(val_spx).join(val_sx5e).join(val_ukx).join(val_vix) 


        #Trainning set

        suffix = 'Label_'+trainfiles_3m[ind].split('_')[0].strip('3M')+'_train_'+str(day)+'d.csv'
        train_label  = pd.read_csv(trainpath+suffix,delimiter=',',index_col=0,usecols=(1,2),names=['date','label'],skiprows=1)

            
        train_oi = pd.read_csv(trainpath+trainfiles_oi[ind],delimiter=',',index_col=0,usecols=(1,2))
        train_3m = pd.read_csv(trainpath+trainfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,5,6))#usecols=(1,2,3,4,5,6))

        train_nky = pd.read_csv(trainpath+trainfiles_indices[0],delimiter=',',index_col=0,usecols=(1,2))
        train_shsz300 = pd.read_csv(trainpath+trainfiles_indices[1],delimiter=',',index_col=0,usecols=(1,2))        
        train_spx = pd.read_csv(trainpath+trainfiles_indices[2],delimiter=',',index_col=0,usecols=(1,2))
        train_sx5e = pd.read_csv(trainpath+trainfiles_indices[3],delimiter=',',index_col=0,usecols=(1,2))
        train_ukx = pd.read_csv(trainpath+trainfiles_indices[4],delimiter=',',index_col=0,usecols=(1,2))
        train_vix = pd.read_csv(trainpath+trainfiles_indices[5],delimiter=',',index_col=0,usecols=(1,2))

        train_data = train_oi.join(train_3m).join(train_nky).join(train_shsz300).join(train_spx).join(train_sx5e).join(train_ukx).join(train_vix) 
        
        all_data = pd.concat([train_data,val_data])
        # print(all_data.isnull().sum()) # Missing Value
        all_data.fillna(method='ffill',inplace=True)
        # print(all_data.isnull().sum()) # Missing Value
        all_data = all_data.join(train_label)

        # Construct new features         
        all_data['Price_diff_1'] = all_data['Close.Price'].diff(1)
        all_data['Price_diff_5'] = all_data['Close.Price'].diff(5)
        all_data['Price_diff_10'] = all_data['Close.Price'].diff(10)
        all_data['Price_diff_15'] = all_data['Close.Price'].diff(15)
        all_data['Price_diff_20'] = all_data['Close.Price'].diff(20)

        data = all_data[-traindata_len-253:] #253 is the length of validation set
        return data


def train_rf(feature,label):
        rf =  RandomForestClassifier(random_state=10,n_estimators=10)
        rf.fit(feature,label)
                
        # y_pred = rf.predict(feature)
        # print ("Accuracy :{}".format(np.mean(label.values==y_pred)))

        # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
        # tree.plot_tree(rf.estimators_[0],
        #        feature_names = feature.columns, 
        #        class_names=['0','1'],
        #        filled = True)
        # plt.show()

        return rf

def val():


    prediction = pd.DataFrame()
    prediction['id'] = []
    prediction['label'] = []

    accuracy = 0    
    for ind in range(6):

        traindata_len = 60 # window_size 
        data = feature_extract(traindata_len,ind=ind)

        window_start = traindata_len +253
        window_end = 253

        valdata_len = 60  # accuracy will achieve 84 when valdata_len = 1, but seems will be data leak...

        flag = 1
        y_pred_all = np.array([])
        
        result = pd.read_csv('result_93.58.csv')
        prefix = valfiles_oi[ind].split('_')[0]+'-validation-60d'
        while(flag):
                if(window_end <= valdata_len):
                         valdata_len =  window_end 
                         flag = 0
                        
                train_data = data[-window_start:-window_end] 
                train_feature = train_data[train_data.columns.difference(['label'])]
                train_label = train_data['label']
                rf = train_rf(train_feature,train_label)

                if(flag):
                        val_data = data[-window_end:-window_end+valdata_len]
                else:
                        val_data = data[-window_end:]
                val_feature = val_data[val_data.columns.difference(['label'])]
        
                y_pred = rf.predict(val_feature)
                y_pred_all = np.append(y_pred_all,y_pred)

                if(flag):
                        val_label  = result.loc[result['id'].str.contains(prefix)][-window_end:-window_end+valdata_len]
                        data.loc[-window_end:-window_end+valdata_len,'label'] = val_label['label'].values #y_pred
                else:
                        val_label  = result.loc[result['id'].str.contains(prefix)][-window_end:]                        
                        data.loc[-window_end:,'label'] = val_label['label'].values #y_pred
                print(np.mean(val_label['label']==y_pred))


                window_start -= valdata_len
                window_end -= valdata_len

        acc = np.mean(result.loc[result['id'].str.contains(prefix)][-253:]['label'].values==y_pred_all)
        accuracy += acc
        print("accuracy: ",acc)
        temp = pd.DataFrame({'id':result[result['id'].str.contains(prefix)]['id'],'label':y_pred_all})

        prediction = prediction.append(temp)
        # break
    print("Average accuracy:",accuracy/6)

    return prediction


if __name__ == "__main__":

        prediction = val()
        
        prediction['label'] = prediction['label'].astype(int)
        prediction.to_csv('result.csv',index=False)
