from test_60d import val as val_60
from test_20d import val as val_20
from test_1d import val as val_1

import pandas as pd
import time

time_start=time.time()

print("Start Now，will take few minutes")


prediction = pd.DataFrame()
prediction['id'] = []
prediction['label'] = []

print("Predicting 60day ....")
prediction = prediction.append( val_60() )

prediction = prediction.append( val_20() )

prediction = prediction.append( val_1() )

prediction['label'] = prediction['label'].astype(int)
prediction.to_csv('result.csv',index=False)


time_end=time.time()
print('Done,time cost:',time_end-time_start,'s')