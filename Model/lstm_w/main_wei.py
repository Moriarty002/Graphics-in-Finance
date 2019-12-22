import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import Tools

from time import sleep
import os


def generate_df_affect_by_n_days(series, n, index=False):
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    clist=['開盤價','收盤價','最高價','最低價','RSI_6','RSI_12','K','D','BIAS','WMR','EMA_12','EMA_26','MACD','psy_6','MTM_6','SAR_6','DM+(DMI)','DM-(DMI)','TR(DMI)','+DI(DMI)','-DI(DMI)','ADX(DMI)','CDP','AH','NH','NL','AL','Trend']
    
    df = pd.DataFrame()
    for i in range(n):
        for j in clist:
            df['%s%d' % (j,i)] = series[j].tolist()[i:-(n - i)]
    df['y'] = series['最高價'].tolist()[n:]
    if index:
        df.index = series.index[n:]
    return df


def readData(column='最高價', n=30, all_too=True, index=False, train_end=-300):
    df_raw, df = Tools.checkCodeInDir('0050')
    df_raw.sort_index(inplace=True)
    df.sort_index(inplace=True)
    clist=['開盤價','收盤價','最高價','最低價','RSI_6','RSI_12','K','D','BIAS','WMR','EMA_12','EMA_26','MACD','psy_6','MTM_6','SAR_6','DM+(DMI)','DM-(DMI)','TR(DMI)','+DI(DMI)','-DI(DMI)','ADX(DMI)','CDP','AH','NH','NL','AL','Trend']
    df_column = df[clist].copy()
    
    df_train, df_test = df_column[:train_end], df_column[train_end - n:]
    
    df_generate_from_df_train = generate_df_affect_by_n_days(df_train, n, index=index)
    if all_too:
        return df_generate_from_df_train, df_column, df.index.tolist()
    return df_generate_from_df_column_train


class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 會用全0的 state
        out = self.out(r_out)
        return out


class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)


n = 30
LR = 0.0001
EPOCH = 10
train_end = -300
clist=['開盤價','收盤價','最高價','最低價','RSI_6','RSI_12','K','D','BIAS','WMR','EMA_12','EMA_26','MACD','psy_6','MTM_6','SAR_6','DM+(DMI)','DM-(DMI)','TR(DMI)','+DI(DMI)','-DI(DMI)','ADX(DMI)','CDP','AH','NH','NL','AL','Trend']
# 數據集建立
df, df_all, df_index = readData('最高價', n=n, train_end=train_end)
plt.plot(df_index, np.array(df_all['最高價']), label='real-data')
df_all = np.array(df_all)


df_numpy = np.array(df)
for i in range(30):
    print(df_numpy[:,i:30:i+810])
os._exit(0)
df_numpy_mean = np.mean(df_numpy,axis=0)

df_numpy_std = np.std(df_numpy,axis=0)

df_numpy = (df_numpy - df_numpy_mean) / df_numpy_std

df_tensor = torch.Tensor(df_numpy)

trainset = TrainSet(df_tensor)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

# rnn = torch.load('rnn.pkl')

rnn = RNN(840)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

for step in range(EPOCH):
    for tx, ty in trainloader:
        output = rnn(torch.unsqueeze(tx, dim=0))
        loss = loss_func(torch.squeeze(output), ty)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()
    print(step, loss)
    if step % 10:
        torch.save(rnn, 'rnn.pkl')
torch.save(rnn, 'rnn.pkl')
#
generate_data_train = []
generate_data_test = []

test_index = len(df_all) + train_end
print(np.shape(df_all))
print(np.shape(df_numpy_mean))
os._exit(0)
df_all_normal = (df_all - df_numpy_mean[:,:-1]) / df_numpy_std[:,:-1]

df_all_normal_tensor = torch.Tensor(df_all_normal)

for i in range(n, len(df_all)):
    x = df_all_normal_tensor[i - n:i]
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
    y = rnn(x)
    if i < test_index:
        generate_data_train.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
    else:
        generate_data_test.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
plt.plot(df_index[n:train_end], generate_data_train, label='generate_train')
plt.plot(df_index[train_end:], generate_data_test, label='generate_test')
plt.legend()
plt.show()
plt.cla()
plt.plot(df_index[train_end:-400], df_all[train_end:-400], label='real-data')
plt.plot(df_index[train_end:-400], generate_data_test[:-400], label='generate_test')
plt.legend()
plt.show()