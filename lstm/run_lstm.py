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
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler


torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class LSTM(nn.Module):
    def __init__(self, n_input, num_layers,n_hidden, n_output):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(n_input,n_hidden,num_layers , batch_first=True)
        
        self.out = nn.Linear(n_hidden, n_output)

        self.sigmoid = nn.Sigmoid()
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
        # x = self.sigmoid(x)
        return x


def loss_mse(pred_y,y,x):
    criteria = nn.MSELoss()
    loss = criteria(pred_y,y)  
    return loss

def loss_2(pred_y , y , x):
    loss = torch.sign(y-x)*(y-pred_y)
    loss = torch.mean(loss)
    return loss

def loss_diff(pred_y , y , x):
    loss = torch.sign(y)*(-pred_y)
    loss = torch.mean(loss)
    return loss

def loss_bce(pred_y , y , x):
    label = torch.where(y-x>0, torch.ones_like(y) , torch.zeros_like(y))
    criteria = nn.BCELoss()
    loss = criteria(pred_y , label)
    return loss

def get_label(y_pre , y_pred):
    label = 1 if y_pred > y_pre else 0;
    return label

def get_label_diff(y_pre,y_pred):
    label = 1 if y_pred>0 else 0;
    return label

def get_label_sigmoid(y_pred):
    label = 1 if y_pred > 0.5 else 0;
    return label


class myDataset(Dataset):
    def __init__(self, filepath, train_day ,day , flag , use_diff):
        # filepath: List[string], path of data ,subscript [0] for training , subscript [1] for validation
        # train_day : int , try to model an LSTM with train_day ,which is specified
        # day: int, try to predict T+day, day is 1 or 20 or 60.
        # flag: string, 'train' or 'test' or 'Validation' , to split the whole dataset
        # use_diff : whether to diff the origin dataset
        self.flag = flag;
        if flag == 'train':
            train_3m = pd.read_csv(filepath[0],delimiter=',',index_col=0,usecols=(1,5))
            val_3m = pd.read_csv(filepath[1],delimiter=',',index_col=0,usecols=(1,5))
            data = train_3m.append(val_3m)
            sc = MinMaxScaler()

            if use_diff:
                data = data.diff(1)
                sc = MaxAbsScaler()


            x =  data[-val_3m.shape[0]-500-train_day+1:-day].values # Price from 2014-1-2 to 2017-12-29
            y = data[-val_3m.shape[0]-500+day:].values  # Price from 2014-1-2+day to 2017-12-29+day
            sc.fit(x)
            x = np.reshape(x,(-1,1)) # For scaling
            x = sc.transform(x)
            x = np.reshape(x, (len(x),1,1))

            y = np.reshape(y,(-1,1)) # For scaling
            y = sc.transform(y)

            # if use_diff:
            #     x = np.diff(x,axis=0)
            #     y = np.diff(y,axis=0)


        elif flag == 'test':
            train_3m = pd.read_csv(filepath[0],delimiter=',',index_col=0,usecols=(1,5))
            val_3m = pd.read_csv(filepath[1],delimiter=',',index_col=0,usecols=(1,5))
            data = train_3m.append(val_3m)
            sc = MinMaxScaler()
            
            if use_diff:
                data = data.diff(1)
                sc = MaxAbsScaler()


            x = data[train_3m.shape[0]-train_day : ].values # Price from 2014-1-2 to 2017-12-29
            y = data[train_3m.shape[0]:].values # used for the previous point
            sc.fit(x)

            x = np.reshape(x,(-1,1)) # For scaling
            x = sc.transform(x)
            x = np.reshape(x, (len(x),1,1))
            
            y = np.reshape(y,(-1,1)) # For scaling
            y = sc.transform(y)

            # if use_diff:
            #     x = np.diff(x,axis=0)
            #     y = np.diff(y,axis=0)


            
        
        
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
          

def train(epochs,model, metal, day, add_epoch , loss_func):
    # metal : what kind of metal
    # day:  1 or 20 or 60.
    L = []
    y1 = np.array([])
    y2 = np.array([])
    best_loss = float('inf')
    epoch = 1
    while epoch < epochs+1:
        model.train()
        train_loss = 0
        for batch_idx, (x,y) in enumerate(train_loader):
            x = x.to(device) #(bs,seq , input)
            y = y.to(device) # (bs,)
            optimizer.zero_grad()
            pred_y = model(x) #(bs,output)

            loss = loss_func(pred_y.squeeze(-1),y,x.squeeze()[:,-1])
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if epoch == epochs:
                y1 = np.append(y1,y.detach().cpu().numpy())
                y2 = np.append(y2,pred_y.detach().cpu().numpy())

        epoch_loss = train_loss / len(train_loader.dataset)
        if epoch % 3 == 0:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, epoch_loss ))
                if add_epoch and epoch == epochs and epoch_loss > 0:
                    epochs += 10
                    
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        L.append(train_loss / len(train_loader.dataset))

        epoch += 1

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
    
    plt.plot(y1,label='real',color='blue')
    plt.plot(y2,label='pred',color='red')
    plt.legend()
    plt.savefig(os.path.join("output/{}_{}day_trainingset_prediction_vs_real.png").format(metal.split('_')[0].strip('3M'), day ))
    plt.close()
    return model 


def test(model, metal, day , label_gen):
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
            label = label_gen(y_pre , pred_y);

            idx = prefix + date.iloc[i]
            test_result = test_result.append({'id':idx , 'label':label},ignore_index=True)
    return test_result

def val(pred,label,prefix):
    
    accuracy = (pred.values==label)['label'].value_counts()
    acc = accuracy[True]/ (accuracy[True] + accuracy[False])
    print(prefix , '的准确率如下：' , acc)

trainpath = 'datasets/compeition_sigir2020/Train/Train_data'
valpath = 'datasets/compeition_sigir2020/Validation/Validation_data'
trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']
valfiles_3m = ['LMEAluminium3M_validation.csv','LMECopper3M_validation.csv','LMELead3M_validation.csv','LMENickel3M_validation.csv','LMETin3M_validation.csv','LMEZinc3M_validation.csv']

#序列长度为
seq_length = 22 
n_hidden = 4
num_layers = 2
epoches = 9
add_epoch = False; # 是否要根据条件添加 epoch
use_diff = False;
train_output = 'output/train_loss/mlp' #命名以 model 进行命名
if not os.path.exists(train_output):
    os.makedirs(train_output)

prediction = pd.DataFrame()
prediction['id'] = []
prediction['label'] = []
result = pd.read_csv('result_93.58.csv')
for ind in range(len(trainfiles_3m)):
    for i in [1,20,60]:
        # The train_data doesn't split training set and test set, so we do it manually by pass a string parameter.
        train_file = os.path.join(trainpath , trainfiles_3m[ind])
        val_file = os.path.join(valpath , valfiles_3m[ind])
        
        train_data  =  myDataset([train_file,val_file], seq_length ,i ,'train' , use_diff)
        test_data  =  myDataset([train_file, val_file], seq_length ,i, 'test' , use_diff)

        train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=False)
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        
        rnn = LSTM(n_input=1, n_hidden=n_hidden, num_layers=num_layers, n_output=1).to(device)
        rnn.init_weight() # 增加初始化
        optimizer = optim.SGD(rnn.parameters(), lr=0.001)

        rnn = train(epochs=epoches, model = rnn, metal=trainfiles_3m[ind], day=i , add_epoch = add_epoch ,loss_func=loss_2 )
        label_pred = test(model=rnn,metal=val_file , day=i ,label_gen = get_label)

        
        prefix = valfiles_3m[ind].split('_')[0].strip('3M')+'-validation-'+str(i)+'d'
        label  = result.loc[result['id'].str.contains(prefix)]
        val(pred=label_pred,label=label,prefix=prefix)

        prediction = prediction.append(label_pred)
        
        # torch.save(mlp,'E:/BaiduNetdiskDownload/compeition_sigir2020/%sday_%s_model'%(i,trainfiles_3m[ind].split('_')[0].strip('3M')))
        

    
prediction['label'] = prediction['label'].astype(int)
accuracy = (prediction.values==result)['label'].value_counts()
acc = accuracy[True]/ (accuracy[True] + accuracy[False])
print('的准确率如下：' , acc)
prediction.to_csv('output/lstm_seq_{}_layers_{}_hidden_{}_epoch_{}__new_loss.csv'.format(seq_length,num_layers , n_hidden , epoches),index=False)
