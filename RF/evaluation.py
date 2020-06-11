import pandas as pd
import numpy as np


for i in [1,20,60]:

    # for ind in['LMEAluminium','LMECopper','LMELead','LMENickel','LMETin','LMEZinc']:

        
        result = pd.read_csv('result_93.58.csv')
        prefix = '-validation-'+str(i)+'d'
        label  = result.loc[result['id'].str.contains(prefix)]

        pred_result = pd.read_csv('result.csv')
        pred_label = pred_result.loc[pred_result['id'].str.contains(prefix)]

        accuracy = np.mean(pred_label['label'].values==label['label'].values)
        print(accuracy)
