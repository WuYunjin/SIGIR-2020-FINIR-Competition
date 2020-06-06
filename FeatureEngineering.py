import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

valpath = 'datasets/compeition_sigir2020/Validation/Validation_data/'
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']

valfiles_oi = ['LMEAluminium_OI_validation.csv','LMECopper_OI_validation.csv','LMELead_OI_validation.csv','LMENickel_OI_validation.csv','LMETin_OI_validation.csv','LMEZinc_OI_validation.csv']

valfiles_indices = ['Indices_NKY Index_validation.csv','Indices_SHSZ300 Index_validation.csv','Indices_SPX Index_validation.csv','Indices_SX5E Index_validation.csv','Indices_UKX Index_validation.csv','Indices_VIX Index_validation.csv']

result = pd.read_csv('result_93.58.csv')

for day in [1,20,60]:
    for ind in range(len(valfiles_3m)):
    
        prefix = valfiles_3m[ind].split('_')[0].strip('3M')+'-validation-'+str(day)+'d'
        label  = result.loc[result['id'].str.contains(prefix)]

            
        val_oi = pd.read_csv(valpath+valfiles_oi[ind],delimiter=',',index_col=0,usecols=(1,2))
        val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6))

        val_nky = pd.read_csv(valpath+valfiles_indices[0],delimiter=',',index_col=0,usecols=(1,2))
        val_shsz300 = pd.read_csv(valpath+valfiles_indices[1],delimiter=',',index_col=0,usecols=(1,2))        
        val_spx = pd.read_csv(valpath+valfiles_indices[2],delimiter=',',index_col=0,usecols=(1,2))
        val_sx5e = pd.read_csv(valpath+valfiles_indices[3],delimiter=',',index_col=0,usecols=(1,2))
        val_ukx = pd.read_csv(valpath+valfiles_indices[4],delimiter=',',index_col=0,usecols=(1,2))
        val_vix = pd.read_csv(valpath+valfiles_indices[5],delimiter=',',index_col=0,usecols=(1,2))

        feature = val_oi.join(val_3m).join(val_nky).join(val_shsz300).join(val_spx).join(val_sx5e).join(val_ukx).join(val_vix) 
        # feature.isna().sum() , indices 有缺失值
        feature.fillna(method='ffill',inplace=True)
        feature.fillna(method='bfill',inplace=True)

        # Construct new features 
        feature['1day_diff'] = feature['Close.Price'].diff(1)

        # Visualize the relation of label and feature.
        for i in range(feature.shape[1]):
            plt.scatter(x=label['label'].values,y=feature.iloc[:,i].values)
            plt.show()
            plt.close()

        break

    break


