import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import GridSearchCV
from sklearn import tree
from talib.abstract import *

valpath = 'datasets/compeition_sigir2020/Validation/Validation_data/'
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']

valfiles_oi = ['LMEAluminium_OI_validation.csv','LMECopper_OI_validation.csv','LMELead_OI_validation.csv','LMENickel_OI_validation.csv','LMETin_OI_validation.csv','LMEZinc_OI_validation.csv']

valfiles_indices = ['Indices_NKY Index_validation.csv','Indices_SHSZ300 Index_validation.csv','Indices_SPX Index_validation.csv','Indices_SX5E Index_validation.csv','Indices_UKX Index_validation.csv','Indices_VIX Index_validation.csv']

result = pd.read_csv('result_93.58.csv')

def val():
        day = 60
        ind = 1
    
        prefix = valfiles_3m[ind].split('_')[0].strip('3M')+'-validation-'+str(day)+'d-'
        label  = result.loc[result['id'].str.contains(prefix)]

            
        val_oi = pd.read_csv(valpath+valfiles_oi[ind],delimiter=',',index_col=0,usecols=(1,2))
        val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)

        val_nky = pd.read_csv(valpath+valfiles_indices[0],delimiter=',',index_col=0,usecols=(1,2))
        val_shsz300 = pd.read_csv(valpath+valfiles_indices[1],delimiter=',',index_col=0,usecols=(1,2))        
        val_spx = pd.read_csv(valpath+valfiles_indices[2],delimiter=',',index_col=0,usecols=(1,2))
        val_sx5e = pd.read_csv(valpath+valfiles_indices[3],delimiter=',',index_col=0,usecols=(1,2))
        val_ukx = pd.read_csv(valpath+valfiles_indices[4],delimiter=',',index_col=0,usecols=(1,2))
        val_vix = pd.read_csv(valpath+valfiles_indices[5],delimiter=',',index_col=0,usecols=(1,2))

        feature = val_oi.join(val_3m).join(val_nky).join(val_shsz300).join(val_spx).join(val_sx5e).join(val_ukx).join(val_vix) 

        # # Construct new features 
        tech_indicators = pd.DataFrame(SMA(val_3m, timeperiod=10))
        tech_indicators.columns = ['sma_10']
        tech_indicators['mom_10'] = pd.DataFrame(MOM(val_3m,10))
        tech_indicators['wma_10'] = pd.DataFrame(WMA(val_3m,10))
        tech_indicators = pd.concat([tech_indicators,STOCHF(val_3m, 
                                          fastk_period=14, 
                                          fastd_period=3)],
                             axis=1)
        period = 20
        tech_indicators['macd'] = pd.DataFrame(MACD(val_3m, fastperiod=12, slowperiod=26)['macd'])
        tech_indicators['rsi'] = pd.DataFrame(RSI(val_3m, timeperiod=14))
        tech_indicators['willr'] = pd.DataFrame(WILLR(val_3m, timeperiod=14))
        tech_indicators['cci'] = pd.DataFrame(CCI(val_3m, timeperiod=14))
        tech_indicators['adosc'] = pd.DataFrame(ADOSC(val_3m, fastperiod=3, slowperiod=10))
        tech_indicators['pct_change'] = ROC(val_3m, timeperiod=period)
        tech_indicators['pct_change'] = tech_indicators['pct_change'].shift(-period)
        tech_indicators['pct_change'] = tech_indicators['pct_change'].apply(lambda x: '1' if x > 0 else '0' if x <= 0 else np.nan)

        tech_indicators = tech_indicators.dropna()

        # # feature.isna().sum() , indices 有缺失值
        feature.fillna(method='ffill',inplace=True)
        feature.fillna(method='bfill',inplace=True)

        # Visualize the relation of label and feature.
        nfrow = 4
        nfcol = 4 
        n_features = tech_indicators.shape[1]
        plt.figure(figsize=(10,10))
        for i in range(n_features):
            plt.subplot(nfrow,nfcol,i+1)
            plt.ylabel(tech_indicators.columns[i])
            plt.scatter(x=label['label'][33:-20].values,y=tech_indicators.iloc[:,i].values)
        plt.subplot(nfrow,nfcol,nfcol*nfrow-nfcol//2)
        plt.xlabel("Validation set")
        plt.show()
        plt.close()


        # rf =  RandomForestClassifier(oob_score=True, random_state=10,verbose=1,max_depth=2,n_estimators=5)
        # rf.fit(feature,label['label'])
                
        # y_pred = rf.predict(feature)
        # print ("Accuracy :{}".format(np.mean(label['label'].values==y_pred)))

        # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
        # tree.plot_tree(rf.estimators_[0],
        #        feature_names = feature.columns, 
        #        class_names=['0','1'],
        #        filled = True)
        # plt.show()
        

if __name__ == "__main__":
    val()


