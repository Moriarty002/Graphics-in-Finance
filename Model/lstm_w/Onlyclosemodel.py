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

def readTrain(stock_num):
    df_raw, df = Tools.checkCodeInDir(stock_num)
    df_raw.sort_index(inplace=True)
    df.sort_index(inplace=True)
    clist=['收盤價']
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

def buildManyToOneModel(shape,nd):
    model = Sequential()
    model.add(LSTM(nd, input_length=shape[1], input_dim=shape[2]))
    # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model

def stock_train(stock_num):
    ndata=28
    df = readTrain(stock_num)
    df_norm = normalize(df)
    X_train_ori, Y_train_ori = buildTrain(df_norm, 30, 1)
    #X_train, Y_train = shuffle(X_train, Y_train)
    #cut the data to test
    X_train,Y_train,x__test,y_test = splitData(X_train_ori, Y_train_ori, 0.95)
    #cut the data to train & prove
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.9)

    #train models
    model = buildManyToOneModel(X_train.shape,ndata)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
    model.save('onlyclose/'+stock_num+'.h5')
    #LOAD
    #from keras.models import load_model
    #model = load_model("./my_model.h5")

    Y_predict=model.predict(X_train)
    Y_val_predict=model.predict(X_val)
    plt.plot(Y_train,label='ans')
    plt.plot(Y_predict,label='predict')
    plt.plot(Y_val,label='ans2')
    plt.plot(Y_val_predict,label='predict2')
    plt.legend()
    plt.savefig('onlyclose/'+stock_num+'_onlyclose.jpg')