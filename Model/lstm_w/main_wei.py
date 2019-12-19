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



# def generate_df_affect_by_n_days(series, n, index=False):
#     if len(series) <= n:
#         raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
#     df = pd.DataFrame()
#     for i in range(n):
#         df['c%d' % i] = series.tolist()[i:-(n - i)]
#     df['y'] = series.tolist()[n:]
#     if index:
#         df.index = series.index[n:]

#     return df

def generate_data_by_n_days(series, n, index=False):
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))

    df_index = series[['RSI_6','RSI_12','K','D','BIAS','WMR','EMA_12','EMA_26','MACD','psy_6','MTM_6','SAR_6','DM+(DMI)','DM-(DMI)','TR(DMI)','+DI(DMI)','-DI(DMI)','ADX(DMI)','Trend']].copy()

    # print(df_index)

    data = []

    for i in range(len(df_index)-30):
        tmp = np.array(df_index.iloc[i:i+n])
        # print(df_index.iloc[i:i+30])
        # sleep(10)
        tmp = tmp.flatten()
        print(tmp)
        os._exit()
        # print(tmp.shape)
        data.append(tmp)
    
    # print(data)

    df = pd.DataFrame(data)

    

    return df


def readData(column='high', n=30, all_too=True, index=False, train_end=-300):
    # df = pd.read_csv("399300.csv", index_col=0)
    df_raw, df = Tools.checkCodeInDir('0050')
    df_raw.sort_index(inplace=True)
    df.sort_index(inplace=True)
    label = df['最高價'].iloc[30:]
    # df.index = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), df.index))
    
    # print(label)
    # os._exit()

    df_all = generate_data_by_n_days(df, n, index=index)
    df_train, df_test = df_all[:train_end], df_all[train_end - n:]

    if all_too:
        return df_train, df_all, df.index.tolist(), label
    return df_train


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
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out)
        return out


class TrainSet(Dataset):
    def __init__(self, data, label):
        # 定义好 image 的路径
        self.data = data.float()
        self.label = label.float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def my_std(df):

    label_numpy = np.array(df)

    label_numpy_mean = np.mean(label_numpy)
    label_numpy_std = np.std(label_numpy)

    label_numpy = (label_numpy - label_numpy_mean) / label_numpy_std
    df_tensor = torch.Tensor(label_numpy)

    return df_tensor


n = 30
INPUT_SIZE = 19*n
LR = 0.0001
EPOCH = 100
train_end = -500
# 数据集建立
df, df_all, df_index, label = readData('最高價', n=n, train_end=train_end)

df_all = np.array(df_all)

# plt.plot(df_index, df_all, label='real-data')

df_tensor = my_std(df)

# print(df_tensor)

# sleep(3000)

label_numpy = np.array(label)

label_numpy_mean = np.mean(label_numpy)
label_numpy_std = np.std(label_numpy)

label_numpy = (label_numpy - label_numpy_mean) / label_numpy_std
label_tensor = torch.Tensor(label_numpy)

trainset = TrainSet(df_tensor, label_tensor)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

# rnn = torch.load('rnn.pkl')

rnn = RNN(INPUT_SIZE)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

for step in range(EPOCH):
    for tx, ty in trainloader:
        output = rnn(torch.unsqueeze(tx, dim=0))
        loss = loss_func(torch.squeeze(output), ty)
        # print(ty)
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

df_all_normal = (df_all - label_numpy_mean) / label_numpy_std
df_all_normal_tensor = torch.Tensor(df_all_normal)


for i in range(n, len(df_all)):
    x = df_all_normal_tensor[i]
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
    y = rnn(x)
    if i < test_index:
        generate_data_train.append(torch.squeeze(y).detach().numpy())
    else:
        generate_data_test.append(torch.squeeze(y).detach().numpy())
        
plt.plot(df_index[30:], label_tensor, label='real-data')
plt.plot(df_index[n+30:train_end], generate_data_train, label='generate_train')
plt.plot(df_index[train_end:], generate_data_test, label='generate_test')
plt.legend()
plt.show()
plt.cla()
plt.plot(df_index[train_end:-400], label[train_end:-400], label='real-data')
plt.plot(df_index[train_end:-400], generate_data_test[:-400], label='generate_test')
plt.legend()
plt.show()