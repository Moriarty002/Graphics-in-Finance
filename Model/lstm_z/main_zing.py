import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


def generate_df_affect_by_n_days(df_in, n, index=False):
    if len(df_in) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(df_in), n))
    df = pd.DataFrame()

    for col in df_in.columns.values.tolist():
        for i in range(n):
            df[col + '%d' % i] = df_in[col].tolist()[i: - (n - i)]


    df['y'] = df_in['收盤價'].tolist()[n:]
    if index:
        df.index = df_in.index[n:]
    return df


def readData(column='收盤價', n=30, all_too=True, index=False, train_end=-300):
    df = pd.read_json("0050_index.json")
    df.sort_index(inplace=True)
    for i in range(len(df)):
        df['日期'].iloc[i]=df['日期'] .iloc[i].replace(df['日期'].iloc[i][0:3],str(int(df['日期'].iloc[i][0:3])+1911))
    df.index = list(map(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d"), df['日期']))
    lis = df.columns.values.tolist()[1:]
    df_generate_from_df_column_train = generate_df_affect_by_n_days(df[:][lis], n, index=index)

    df_generate_from_df_column_train.to_json('df_generate_from_df_column_train.json')

    if all_too:
        return df_generate_from_df_column_train[:train_end],df_generate_from_df_column_train, df.index.tolist(),df['收盤價']
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
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out)
        return out


class TrainSet(Dataset):
    def __init__(self, data):
        # 定义好 image 的路径
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


n = 30
LR = 0.0001
EPOCH = 100
train_end = -500
# 数据集建立
df, df_all, df_index ,yaxis= readData('收盤價', n=n, train_end=train_end)

print (df)


df_all = np.array(df_all)

plt.plot(df_index,yaxis, label='real-data')

df_numpy = np.array(df)

df_numpy_mean = np.mean(df_numpy)
df_numpy_std = np.std(df_numpy)

df_numpy = (df_numpy - df_numpy_mean) / df_numpy_std
df_tensor = torch.Tensor(df_numpy)

trainset = TrainSet(df_tensor)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

rnn = torch.load('rnn.pkl')
'''
rnn = RNN(1020)
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
'''
generate_data_train = []
generate_data_test = []

test_index = len(df_all) + train_end

df_all_normal = (df_all - df_numpy_mean) / df_numpy_std
df_all_normal_tensor = torch.Tensor(df_all_normal)

print(df_all_normal_tensor)
print(df_tensor)

for i in range(n, len(df_all)):
    x = df_all_normal_tensor[i, :-1].float()
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
    y = rnn(x)
    if i < test_index:
        generate_data_train.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
    else:
        generate_data_test.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)


plt.plot(df_index[n+30:train_end], generate_data_train, label='generate_train')
plt.plot(df_index[train_end:], generate_data_test, label='generate_test')
plt.legend()
plt.show()
plt.cla()
plt.plot(df_index[train_end:-400], df_all[train_end:-400], label='real-data')
plt.plot(df_index[train_end:-400], generate_data_test[:-400], label='generate_test')
plt.legend()
plt.show()

