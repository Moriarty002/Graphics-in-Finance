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

def readTrain(df_index):
    df_index.sort_index(inplace=True)
    clist=['開盤價','收盤價','最高價','最低價','RSI_6','RSI_12','K','D','BIAS','WMR','EMA_12','EMA_26','MACD','psy_6','MTM_6','SAR_6','DM+(DMI)','DM-(DMI)','TR(DMI)','+DI(DMI)','-DI(DMI)','ADX(DMI)','CDP','AH','NH','NL','AL','Trend']
    df_column = df_index[clist].copy()
    return df_column

def normalize(train):
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm,train.apply(lambda x:np.mean(x)),train.apply(lambda x:(np.max(x) - np.min(x)))

def denormalize(array,mean,diff):
    array_new=(array*diff)+mean
    return array_new
    
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

def buildManyToOneModel(shape,nd):
    model = Sequential()
    model.add(LSTM(nd, input_length=shape[1], input_dim=shape[2]))
    # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model

def selfMSE(A,B):
    loss=np.mean(((A-B)*(A-B)))
    return loss

def stock_train(stock_num,df_index,train=True):
    ndata=28
    df = readTrain(df_index)
    df_norm,dfmean,dfdiff = normalize(df)
    X_train_ori, Y_train_ori = buildTrain(df_norm, 30, 1)
    #X_train, Y_train = shuffle(X_train, Y_train)
    #cut the data to test
    X_train,Y_train,X_test,Y_test = splitData(X_train_ori, Y_train_ori, 0.95)
    #cut the data to train & prove
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.9)
    if(train):
        #train models
        model = buildManyToOneModel(X_train.shape,ndata)
        callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
        model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
        model.save('withidx/'+stock_num+'.h5')
    else:
        #LOAD
        from keras.models import load_model
        model = load_model('withidx/'+stock_num+'.h5')

    Y_predict=model.predict(X_train_ori)
    
    Y_train_ori=Y_train_ori.flatten()
    Y_predict=Y_predict.flatten()
    a=Y_train.size
    b=Y_val.size
    c=Y_test.size
    print('WD')
    print(selfMSE(Y_predict[0:a+b],Y_train_ori[0:a+b]))
    
    Y_train_ori=denormalize(Y_train_ori,dfmean['收盤價'],dfdiff['收盤價'])
    Y_predict=denormalize(Y_predict,dfmean['收盤價'],dfdiff['收盤價'])
    
    print(selfMSE(Y_predict[0:a+b],Y_train_ori[0:a+b]))
    
    plt.plot(range(a),Y_train_ori[0:a],label='ans_train',lw=3)
    plt.plot(range(a,a+b),Y_train_ori[a:a+b],label='ans_val',lw=3)
    plt.plot(range(a+b,a+b+c),Y_train_ori[a+b:a+b+c],label='ans_test',lw=3)
    plt.plot(range(Y_predict.size),Y_predict,label='predict',lw=0.5)
    plt.legend()
    plt.savefig('withidx/'+stock_num+'_withidx.png',dpi=600)
    plt.cla()
    
'''
Y_train.size    a-->r
Y_val.size      b-->g
Y_test.size     c-->b

X_train_ori
Y_train_ori
Y_predict
'''





