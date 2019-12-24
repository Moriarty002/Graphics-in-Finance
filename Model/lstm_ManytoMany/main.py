import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import datetime
import Tools
from time import sleep
import os

def readTrain():
    df_raw, df = Tools.checkCodeInDir('0050')
    df_raw.sort_index(inplace=True)
    df.sort_index(inplace=True)
    clist=['開盤價','收盤價','最高價','最低價','RSI_6','RSI_12','K','D','BIAS','WMR','EMA_12','EMA_26','MACD','psy_6','MTM_6','SAR_6','DM+(DMI)','DM-(DMI)','TR(DMI)','+DI(DMI)','-DI(DMI)','ADX(DMI)','CDP','AH','NH','NL','AL','Trend']
    df_column = df[clist].copy()
    return df_column

def normalize(train):
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm
    
def buildTrain(train, pastDay=30, futureDay=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]['收盤價']))
    return np.array(X_train), np.array(Y_train)

def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def splitData(X,Y,rate):
    X_train = X[:int(X.shape[0]*rate)]
    Y_train = Y[:int(Y.shape[0]*rate)]
    X_val = X[int(X.shape[0]*rate):]
    Y_val = Y[int(Y.shape[0]*rate):]
    return X_train, Y_train, X_val, Y_val


def buildManyToManyModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    # output shape: (5, 1)
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model

ndata=28
df = readTrain()
df_norm = normalize(df)
X_train, Y_train = buildTrain(df_norm, 30, 30)
#X_train, Y_train = shuffle(X_train, Y_train)
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.9)

# from 2 dimmension to 3 dimension
Y_train = Y_train[:,:,np.newaxis]
Y_val = Y_val[:,:,np.newaxis]



for i in range(30):
    for k in range(30):
        plt.plot(k+i,Y_train[i][k][0], label='ans'+str(i))
    
plt.legend()
plt.show()
plt.cla()
#train models
print(X_train.shape)
'''
model = buildManyToManyModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
model.save('my_model.h5')
'''
#LOAD

'''
from keras.models import load_model
model = load_model("./my_model.h5")

Y_predict=model.predict(X_train)
Y_val_predict=model.predict(X_val)


for i in range(Y_predict.shape[0]):
    plt.plot(Y_predict[i], label='predict')

plt.legend()
plt.show()
plt.cla()

'''