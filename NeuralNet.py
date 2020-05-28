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

class MLP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden//2)

        self.out = torch.nn.Linear(n_hidden//2, n_output)  # output layer

    def init_weight(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform(param)
            else:
                nn.init.constant(param, 0.1)
            pass

    def forward(self, x):

        x = torch.relu(self.hidden_1(x))  # activation function for hidden layer
        

        x = torch.relu(self.hidden_2(x))  # activation function for hidden layer

        x = self.out(x)

        return x


def loss_function(pred_y,y):
    criteria = nn.MSELoss()
    loss = criteria(pred_y,y)  
    return loss


class myDataset(Dataset):
    def __init__(self, filepath,day,flag):
        # filepath: string, path of data
        # day: int, try to predict T+day, day is 1 or 20 or 60.
        # flag: string, 'train' or 'test' or 'Validation' , to split the whole dataset
        data = pd.read_csv(filepath,delimiter=',',index_col=0,usecols=(1,5))  # Only use Close.Price for now.

        if flag == 'train':
            x =  data[-1011:-252].values # Price from 2014-1-2 to 2016-12-30
            y = data[-1011+day:-252+day].values  # Price from 2014-1-2+day to 2016-12-30+day


            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler()
            x = np.reshape(x,(-1,1)) # For scaling
            x = sc.fit_transform(x)
            x = np.reshape(x, (len(x),1,1))

            y = np.reshape(y,(-1,1)) # For scaling
            y = sc.fit_transform(y)

        elif flag == 'test':
            x = data[-252:-day].values  # Price from 2017-1-3 to 2017-12-29 - day.
            y = data[-252+day:].values  # Price from 2017-1-3 + day to 2017-12-29 .

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
   
    def __getitem__(self, idx):
        
        x = torch.tensor(self.x[idx]).float() # Price at T day
        y = torch.tensor(self.y[idx]).float() # Price at T+self.day day
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
        if epoch % 10 == 0:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, epoch_loss ))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        L.append(train_loss / len(train_loader.dataset))

    import matplotlib.pyplot as plt 
    plt.plot(np.array(L))
    plt.savefig(os.path.join(train_output , "{}_{}day_hidden_size({})_mse({}).png".format(metal.split('_')[0].strip('3M'), day ,4 , best_loss))) #文件命名以 hyperparameter_metrice 形式
    plt.close()

    plt.plot(y1,label='real',color='blue')
    plt.plot(y2,label='pred',color='red')
    plt.legend()
    plt.savefig(os.path.join("output/{}_{}day_trainingset_prediction_vs_real.png").format(metal.split('_')[0].strip('3M'), day ))
    plt.close()
    return model 


def test(model, metal, day):
    model.eval()
    test_loss = 0
    y1 = np.array([])
    y2 = np.array([])
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(test_loader):
            x = x.to(device)
            pred_y = model(x)
            loss = loss_function(pred_y,y)
            test_loss += loss
                
            y1 = np.append(y1,y.numpy())
            y2 = np.append(y2,pred_y.numpy())
        print('test set Average loss: {:.4f}'.format(test_loss / len(test_loader.dataset)))
            
        import matplotlib.pyplot as plt 
        plt.plot(y1,label='real',color='blue')
        plt.plot(y2,label='pred',color='red')
        plt.legend()
        plt.savefig(os.path.join("output/{}_{}day_testset_prediction_vs_real.png").format(metal.split('_')[0].strip('3M'), day ))
        plt.close()



torch.manual_seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainpath = 'datasets/compeition_sigir2020/Train/Train_data'

trainfiles_3m = ['LMEAluminium3M_train.csv','LMECopper3M_train.csv','LMELead3M_train.csv','LMENickel3M_train.csv','LMETin3M_train.csv','LMEZinc3M_train.csv']

train_output = 'output/train_loss/mlp' #命名以 model 进行命名
if not os.path.exists(train_output):
    os.makedirs(train_output)

for ind in range(len(trainfiles_3m)):
    

    for i in [1, 20, 60]:
        # The train_data doesn't split training set and test set, so we do it manually by pass a string parameter.
        train_data  =  myDataset(os.path.join(trainpath , trainfiles_3m[ind]),i,'train')
        test_data  =  myDataset(os.path.join(trainpath , trainfiles_3m[ind]),i,'test')

        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        
        mlp = MLP(n_feature=1 ,n_hidden=4,n_output=1).to(device)
        mlp.init_weight() # 增加初始化
        optimizer = optim.SGD(mlp.parameters(), lr=0.001)

        mlp = train(epochs=500, model = mlp, metal=trainfiles_3m[ind], day=i)
        test(model = mlp, metal=trainfiles_3m[ind], day=i)
        # torch.save(mlp,'E:/BaiduNetdiskDownload/compeition_sigir2020/%sday_%s_model'%(i,trainfiles_3m[ind].split('_')[0].strip('3M')))
        

        # break
    break






    











































# for ind in range(len(files_oi)):
    

#     val_oi = pd.read_csv(valpath+valfiles_oi[ind],delimiter=',',index_col=0,usecols=(1,2))
#     val_3m = pd.read_csv(valpath+valfiles_3m[ind],delimiter=',',index_col=0,usecols=(1,2,3,4,5,6))
#     val_data = val_oi.join(val_3m) # 6-dimension


#     pred = f(val_data.values)
#     prefix = valfiles_oi[ind].split('_')[0]+'-validation-1d-'
#     val_index = prefix + val_oi.index
#     temp = pd.DataFrame({'id':val_index,'label':np.array(pred)})
#     prediction = prediction.append(temp)

#     pred = f(val_data.values)
#     prefix = valfiles_oi[ind].split('_')[0]+'-validation-20d-'
#     val_index = prefix + val_oi.index
#     temp = pd.DataFrame({'id':val_index,'label':np.array(pred)})
#     prediction = prediction.append(temp)

#     pred = f(val_data.values)
#     prefix = valfiles_oi[ind].split('_')[0]+'-validation-60d-'
#     val_index = prefix + val_oi.index
#     temp = pd.DataFrame({'id':val_index,'label':np.array(pred)})
#     prediction = prediction.append(temp)

# prediction['label'] = prediction['label'].astype(int)