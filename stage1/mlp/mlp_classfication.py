# -*- coding: UTF-8 -*-
# author : yunjin zhaolong
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch.utils.data import Dataset,DataLoader
from torch import nn, optim
import pandas as pd
import numpy as np
import os
from talib.abstract import *



torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

class MLP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.1):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden//2)
        self.bn2 = torch.nn.BatchNorm1d(n_hidden//2)

        self.hidden_3 = torch.nn.Linear(n_hidden//2, n_hidden//4)  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(n_hidden//4)

        self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)

        self.out = torch.nn.Linear(n_hidden//8, n_output)  # output layer


    def init_weight(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform(param)
            else:
                nn.init.constant(param, 0.1)
            pass


    def forward(self, x):
        x = torch.relu(self.hidden_1(x))  # activation function for hidden layer
        x = self.bn1(x)
        x = torch.relu(self.hidden_2(x))  # activation function for hidden layer
        x = self.bn2(x)
        x = torch.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.bn3(x)
        x = torch.relu(self.hidden_4(x))  # activation function for hidden layer
        x = self.bn4(x) #self.dropout(self.bn4(x))
        x = self.out(x)
        return x


def loss_function(pred_y,y):
    criteria = torch.nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()
    loss = criteria(pred_y,y) 
    return loss


class myDataset(Dataset):
    def __init__(self, filepath,labelpath,flag):
        # filepath: string, path of data
        # flag: string, 'train' or 'Validation' , to split the whole dataset
        
        data = pd.read_csv(filepath,delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)
        label = pd.read_csv(labelpath,index_col=0,usecols=(1,2),names=['Index','label'],skiprows=1)
        # Construct new features   
        data['sma_10'] = pd.DataFrame(SMA(data, timeperiod=10))
        data['mom_10'] = pd.DataFrame(MOM(data,10))
        data['wma_10'] = pd.DataFrame(WMA(data,10))
        data = pd.concat([data,STOCHF(data, 
                                          fastk_period=14, 
                                          fastd_period=3)],
                             axis=1)
 
        data['macd'] = pd.DataFrame(MACD(data, fastperiod=12, slowperiod=26)['macd'])
        data['rsi'] = pd.DataFrame(RSI(data, timeperiod=14))
        data['willr'] = pd.DataFrame(WILLR(data, timeperiod=14))
        data['cci'] = pd.DataFrame(CCI(data, timeperiod=14))
        
        data['pct_change_20'] = ROC(data, timeperiod=20)
        data['pct_change_30'] = ROC(data, timeperiod=30)
        data['pct_change_60'] = ROC(data, timeperiod=60)
        data.dropna(inplace=True)

        if flag == 'train':
            # Don't fit the MinMaxScaler with Validation set, which causes data leak.  
            traindata_len = 1000
            train_data = data[-traindata_len:]
            x =  sc.fit_transform(train_data.values)

            y = label['label'][-traindata_len:].values



        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
   
    def __getitem__(self, idx):
        
        x = torch.tensor(self.x[idx]).float() 
        y = torch.tensor([self.y[idx]]).float() 
        return x,y  

def train(epochs,model, metal, day):
    # metal : what kind of metal
    # day:  1 or 20 or 60.
    L = []
    y1 = np.array([])
    y2 = np.array([])
    best_loss = float('inf')
    for epoch in range(1, epochs +1):
        model.train()
        train_loss = 0
        for batch_idx, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device) # put y to the same device
            optimizer.zero_grad()
            pred_y = model(x)

            loss = loss_function(pred_y,y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if epoch == epochs:
                y1 = np.append(y1,y.detach().numpy())
                y2 = np.append(y2,pred_y.detach().numpy())

        epoch_loss = train_loss / len(train_loader.dataset)
        if epoch % 20 == 0:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, epoch_loss ))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        L.append(train_loss / len(train_loader.dataset))

    import matplotlib.pyplot as plt 
    plt.plot(np.array(L))
    plt.savefig(os.path.join(train_output , "{}_{}day_hidden_size({})_BCE({}).png".format(metal.split('_')[0].strip('3M'), day ,16 , best_loss))) #文件命名以 hyperparameter_metrice 形式
    plt.close()

    return model 


def val(model,data,label):
    
            # Construct new features   
    data['sma_10'] = pd.DataFrame(SMA(data, timeperiod=10))
    data['mom_10'] = pd.DataFrame(MOM(data,10))
    data['wma_10'] = pd.DataFrame(WMA(data,10))
    data = pd.concat([data,STOCHF(data, 
                                          fastk_period=14, 
                                          fastd_period=3)],
                             axis=1)
 
    data['macd'] = pd.DataFrame(MACD(data, fastperiod=12, slowperiod=26)['macd'])
    data['rsi'] = pd.DataFrame(RSI(data, timeperiod=14))
    data['willr'] = pd.DataFrame(WILLR(data, timeperiod=14))
    data['cci'] = pd.DataFrame(CCI(data, timeperiod=14))
        
    data['pct_change_20'] = ROC(data, timeperiod=20)
    data['pct_change_30'] = ROC(data, timeperiod=30)
    data['pct_change_60'] = ROC(data, timeperiod=60)
    data.dropna(inplace=True)

    data = data[-253:]
    pred_price = pd.DataFrame([],index=data.index,columns=['pred_label']) 
  

    for ind,x in enumerate(data.values):        
        model.eval()
        with torch.no_grad():
            x = sc.transform(x.reshape(1,-1))
            x = torch.tensor(x).float().to(device)
            pred_y = torch.sigmoid(model(x)).numpy()
            
            pred_price['pred_label'].iloc[ind] = pred_y
        
        
        # x at T includes the true price at T-1, so we have a new pair (x,y) sample to update model.
        # Note that it is not a leak.
        # if(ind>0):
        #     model.train()
        #     new_x = data.values[ind-1]
        #     new_x = sc.transform(new_x.reshape(1,-1))
        #     new_x = torch.tensor(new_x).float().to(device)

        #     new_y = sc.transform((data.values[ind]).reshape(1,-1))[0,3]
        #     new_y = torch.tensor(new_y).float().to(device)

        #     optimizer.zero_grad()
        #     y1 = model(new_x)
        #     loss = loss_function(y1,new_y)
        #     loss.backward()
        #     optimizer.step()


    print("max(pred_price['pred_label']):",max(pred_price['pred_label']))
    for thrsh in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:                
        pred_price['pred_label_{}'.format(thrsh)] = pred_price['pred_label'].apply(lambda x: 1 if x>thrsh else 0)

            
        accuracy = np.mean(pred_price['pred_label_{}'.format(thrsh)].values==label['label'].values)
        print("thresh:{} ,{}".format(thrsh,accuracy))



trainpath = 'datasets/compeition_sigir2020/Train/Train_data'
valpath = 'datasets/compeition_sigir2020/Validation/Validation_data'
trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']
labelfiles = ['Label_LMEAluminium_train_1d.csv','Label_LMECopper_train_1d.csv','Label_LMELead_train_1d.csv','Label_LMENickel_train_1d.csv','Label_LMETin_train_1d.csv','Label_LMEZinc_train_1d.csv']
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']

train_output = 'output/train_loss/mlp' #命名以 model 进行命名
if not os.path.exists(train_output):
    os.makedirs(train_output)

for ind in range(len(trainfiles_3m)):
    

    for i in [1]:
        train_data  =  myDataset(os.path.join(trainpath, trainfiles_3m[ind]),os.path.join(trainpath,labelfiles[ind]),'train')

        train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=False)
        
        mlp = MLP(n_feature=17 ,n_hidden=24,n_output=1).to(device)
        mlp.init_weight() # 增加初始化
        optimizer = optim.Adam(mlp.parameters(), lr=0.0005)

        mlp = train(epochs=1000, model = mlp, metal=trainfiles_3m[ind], day=i)

        
        Valdata = pd.read_csv(os.path.join(valpath, valfiles_3m[ind]),delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)
        temp = pd.read_csv(os.path.join(trainpath, trainfiles_3m[ind]),delimiter=',',index_col=0,usecols=(1,2,3,4,5,6),names=['Index','open','high','low','close','volume'],skiprows=1)
        all_data = pd.concat([temp,Valdata])  # Add training data for constructing features of validation data

        result = pd.read_csv('result_93.58.csv')
        prefix = valfiles_3m[ind].split('_')[0].strip('3M')+'-validation-'+str(i)+'d'
        label  = result.loc[result['id'].str.contains(prefix)]
        val(model=mlp,data=all_data,label=label)
        # torch.save(mlp,'E:/BaiduNetdiskDownload/compeition_sigir2020/%sday_%s_model'%(i,trainfiles_3m[ind].split('_')[0].strip('3M')))
        

        # break
    break
