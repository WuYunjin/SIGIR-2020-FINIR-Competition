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


torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

class LSTM(nn.Module):
    def __init__(self, n_input, num_layers,n_hidden, n_output):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(n_input,n_hidden,num_layers , batch_first=True)
        
        self.out = nn.Linear(n_hidden, n_output)
    def init_weight(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform(param)
            else:
                nn.init.constant(param, 0.1)
            pass

    def forward(self, x):
        '''
        x #(batch_size , seq , n_input)
        '''
        output ,(cn ,hn) = self.rnn(x)
        x = torch.relu(output)  # activation function for hidden layer
        x = self.out(x)[:,-1]

        return x


def loss_function(pred_y,y):
    criteria = nn.MSELoss()
    loss = criteria(pred_y,y)  
    return loss


class myDataset(Dataset):
    def __init__(self, filepath, train_day ,day , flag):
        # filepath: List[string], path of data ,subscript [0] for training , subscript [1] for validation
        # train_day : int , try to model an LSTM with train_day ,which is specified
        # day: int, try to predict T+day, day is 1 or 20 or 60.
        # flag: string, 'train' or 'test' or 'Validation' , to split the whole dataset
        self.flag = flag;
        if flag == 'train':
            train_3m = pd.read_csv(filepath[0],delimiter=',',index_col=0,usecols=(1,5))
            val_3m = pd.read_csv(filepath[1],delimiter=',',index_col=0,usecols=(1,5))
            data = train_3m.append(val_3m)

            x =  data[-val_3m.shape[0]-1011-train_day+1:-day].values # Price from 2014-1-2 to 2017-12-29
            y = data[-val_3m.shape[0]-1011+day:].values  # Price from 2014-1-2+day to 2017-12-29+day
            sc.fit(x)
            x = np.reshape(x,(-1,1)) # For scaling
            x = sc.transform(x)
            x = np.reshape(x, (len(x),1,1))

            y = np.reshape(y,(-1,1)) # For scaling
            y = sc.transform(y)

        elif flag == 'test':
            train_3m = pd.read_csv(filepath[0],delimiter=',',index_col=0,usecols=(1,5))
            val_3m = pd.read_csv(filepath[1],delimiter=',',index_col=0,usecols=(1,5))
            data = train_3m.append(val_3m)
            
            x = data[train_3m.shape[0]-train_day : ].values # Price from 2014-1-2 to 2017-12-29
            y = data[train_3m.shape[0]:].values # used for the previous point
            sc.fit(x)

            x = np.reshape(x,(-1,1)) # For scaling
            x = sc.transform(x)
            x = np.reshape(x, (len(x),1,1))
            
            y = np.reshape(y,(-1,1)) # For scaling
            y = sc.transform(y)
            
            

        
        else:
            print('输入的flag不符合逻辑')
            exit()

        self.train_day = train_day
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
   
    def __getitem__(self, idx):
        if self.flag == 'train':
            x = torch.tensor(self.x[idx:idx+self.train_day]).float().squeeze(-1) # Price at T day
            y = torch.tensor(self.y[idx]).float().squeeze(-1) # Price at T+self.day day
            return x,y
        else:
            x = torch.tensor(self.x[idx:idx+self.train_day]).float().squeeze(-1) # Price at T day
            y_pre = torch.tensor(self.y[idx]).float().squeeze(-1) # Price for the previous
            return x ,y_pre
          

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
            x = x.to(device) #(bs,seq , input)
            y = y.to(device) # (bs,)
            optimizer.zero_grad()
            pred_y = model(x) #(bs,output)

            loss = loss_function(pred_y.squeeze(-1),y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if epoch == epochs:
                y1 = np.append(y1,y.detach().cpu().numpy())
                y2 = np.append(y2,pred_y.detach().cpu().numpy())

        epoch_loss = train_loss / len(train_loader.dataset)
        if epoch % 10 == 0:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, epoch_loss ))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        L.append(train_loss / len(train_loader.dataset))

    import matplotlib.pyplot as plt 

    plt.plot(y1,label='real',color='blue')
    plt.plot(y2,label='pred',color='red')
    plt.legend()
    plt.savefig(os.path.join("output/{}_{}day_trainingset_prediction_vs_real.png").format(metal.split('_')[0].strip('3M'), day ))
    plt.close()
    return model 


def test(model, metal, day):
    model.eval()
    test_result = pd.DataFrame();
    test_result['id'] = []
    test_result['label'] = []
    prefix = valfiles_3m[ind].split('_')[0].strip('3M')+'-validation-'+str(day)+'d-'

    date = pd.read_csv(metal,index_col=0,sep=',').iloc[:,0]
    with torch.no_grad():
        for i, (x,y_pre) in enumerate(test_loader):
            x = x.to(device)
            pred_y = model(x)
            pred_y = pred_y.squeeze(-1).cpu().numpy()[-1]
            y_pre = y_pre.cpu().numpy()[-1]
            label = 1 if pred_y > y_pre else 0;
            idx = prefix + date.iloc[i]
            test_result = test_result.append({'id':idx , 'label':label},ignore_index=True)
    return test_result
            

def val(model,data,day,label):
    
    pred_price = pd.DataFrame([],index=data.index,columns=['price'])
    temp = np.array([])


    for ind,x in enumerate(data.values):        
        model.eval()
        with torch.no_grad():
            
            x = torch.tensor(x).float().to(device)
            pred_y = model(x).numpy()
            if( ind+day >= len(data.values)): 
                temp = np.append(temp,pred_y)
            else:
                pred_price['price'].iloc[ind+day] = pred_y
        
        
        # Update the model with the new instance.
        if( ind+day < len(data.values)): 
            model.train()
            new_x = data.values[ind]
            new_x = np.reshape(new_x,(-1,1))
            new_x = sc.transform(new_x)
            new_x = torch.tensor(new_x).float().to(device)

            new_y = data.values[ind+day]
            new_y = np.reshape(new_y,(-1,1)) # For scaling
            new_y = sc.transform(new_y)
            new_y = torch.tensor(new_y).float().to(device)

            optimizer.zero_grad()
            y1 = model(new_x)
            loss = loss_function(y1,new_y)
            loss.backward()
            optimizer.step()


                
    pred_price = pred_price.append(pd.DataFrame(temp,columns=['price']))[day:]
    


    pred_label = pd.DataFrame([],index=data.index[day:],columns=['label'])

    for ind,x in enumerate(pred_price['price'][:-day].values):
        if( x < pred_price['price'].iloc[ind+day] ):
            pred_label['label'].iloc[ind] = 1
        else:
            pred_label['label'].iloc[ind] = 0

        
    accuracy = (pred_label.values==label)['label'].value_counts()
    print(accuracy)



trainpath = 'datasets/compeition_sigir2020/Train/Train_data'
valpath = 'datasets/compeition_sigir2020/Validation/Validation_data'
trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']

#序列长度为
seq_length = 22 
n_hidden = 4
num_layers = 2
epoches = 500
train_output = 'output/train_loss/mlp' #命名以 model 进行命名
if not os.path.exists(train_output):
    os.makedirs(train_output)

prediction = pd.DataFrame()
prediction['id'] = []
prediction['label'] = []
for ind in range(len(trainfiles_3m)):
    for i in [1,20,60]:
        # The train_data doesn't split training set and test set, so we do it manually by pass a string parameter.
        train_file = os.path.join(trainpath , trainfiles_3m[ind])
        val_file = os.path.join(valpath , valfiles_3m[ind])
        
        train_data  =  myDataset([train_file,val_file], seq_length ,i ,'train')
        test_data  =  myDataset([train_file, val_file], seq_length ,i, 'test')

        train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=False)
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        
        rnn = LSTM(n_input=1, n_hidden=n_hidden, num_layers=num_layers, n_output=1).to(device)
        rnn.init_weight() # 增加初始化
        optimizer = optim.SGD(rnn.parameters(), lr=0.001)

        rnn = train(epochs=epoches, model = rnn, metal=trainfiles_3m[ind], day=i)
        label_pred = test(model=rnn,metal=val_file , day=i)
        prediction = prediction.append(label_pred)
        
        # torch.save(mlp,'E:/BaiduNetdiskDownload/compeition_sigir2020/%sday_%s_model'%(i,trainfiles_3m[ind].split('_')[0].strip('3M')))
        

        # break
    # break
prediction = pd.read_csv('output/lstm_layers_2_hidden_4.csv')
prediction['label'] = prediction['label'].astype(int)
prediction.to_csv('output/lstm_layers_{}_hidden_{}_int.csv'.format(num_layers , n_hidden),index=False)
# prediction.to_csv('output/lstm_layers_{}_hidden_{}.csv'.format(num_layers , n_hidden),index=False)