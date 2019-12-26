# 撈股票
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import talib
import mpl_finance as mpf
import seaborn as sns
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import index_generate
import csv
import time
import os
# plt.rcParams['font.family']='SimHei' # 顯示中文('SimHei' for MacOS)
from sklearn import preprocessing


global df1

def checkCodeInDir(stockcode):  # 查詢是否存在，存在回傳1 不存在回傳0 並建立
    PATH_TO_STOCKDATA = './stock_data'
    raw_target = stockcode + '.json'

    PATH_input_json = './stock_data/'+stockcode+'.json'

    PATH_TO_STOCKDATA_INDEX = './stock_data_index'
    index_target = stockcode + '_index.json'
    
    rawInFile = 0
    indexInFile = 0
    df_raw = pd.DataFrame()
    df_index = pd.DataFrame()
    
    for root, dirs, files in os.walk(PATH_TO_STOCKDATA):
        if raw_target in files:
            print('found data ['+ raw_target + ']at : '+root)
            rawInFile = 1
            df_raw=pd.read_json(PATH_input_json)

    if(rawInFile==0):
        print("["+ raw_target + "] is not be found ,crawling data from ''https://www.twse.com.tw/exchangeReport/STOCK_DAY?'',\nplease wait ")
        df_raw=get_stockdata(stockcode)

    for root, dirs, files in os.walk(PATH_TO_STOCKDATA_INDEX):
        if index_target in files:
            print('found data ['+ index_target + ']at : '+root)
            indexInFile = 1

    if (indexInFile == 0):
        print(" [" + index_target + "] is not be found ,stock_index_generator is running,\nplease wait ")
        df_index=index_generate.stock_index_generator(df_raw,stockcode)
    else:
        PATH_TO_STOCKDATA_INDEX = './stock_data_index/'
        index_target = stockcode+'_index.json'
        df_index = pd.read_json(PATH_TO_STOCKDATA_INDEX+index_target)
    return df_raw,df_index


def get_stockdata(stockcode):  # 接從網路抓股票代碼 並儲存在./stock_data/XXXXX.json 裡面

    date = "2010/01/01"
    inday = datetime.datetime.strptime(date, "%Y/%m/%d")
    ago = [(inday + relativedelta(months=+i)).strftime('%Y%m%d')
            for i in range(131)]
    month = ago[0]

    data = {
        'response': 'json',
        'date': month,
        'stockNo': stockcode,
    }
    dres = requests.get(
        'https://www.twse.com.tw/exchangeReport/STOCK_DAY?', params=data)
    result = json.loads(dres.text)
    df1 = pd.DataFrame(result['data'])
    df1.columns = result['fields']
    print(month)
    time.sleep(3)

    for month in ago[1:119]:
        data = {
            'response': 'json',
            'date': month,
            'stockNo': stockcode,
        }
        dres = requests.get(
            'https://www.twse.com.tw/exchangeReport/STOCK_DAY?', params=data)
        result = json.loads(dres.text)
        tmp = pd.DataFrame(result['data'])
        tmp.columns = result['fields']
        df1 = df1.append(tmp).reset_index(drop=True)
        print(month)
        time.sleep(3)

    df1['開盤價'] = df1['開盤價'].str.replace(
        ',', '').astype(dtype=float, errors='ignore')
    df1['最高價'] = df1['最高價'].str.replace(
        ',', '').astype(dtype=float, errors='ignore')
    df1['最低價'] = df1['最低價'].str.replace(
        ',', '').astype(dtype=float, errors='ignore')
    df1['收盤價'] = df1['收盤價'].str.replace(
        ',', '').astype(dtype=float, errors='ignore')
    df1['成交股數'] = df1['成交股數'].str.replace(
        ',', '').astype(dtype=float, errors='ignore')
    df1['成交金額'] = df1['成交金額'].str.replace(
        ',', '').astype(dtype=float, errors='ignore')
    df1['成交筆數'] = df1['成交筆數'].str.replace(
        ',', '').astype(dtype=float, errors='ignore')
    df1['漲跌價差'] = df1['漲跌價差'].str.replace('+', '')
    df1['漲跌價差'] = df1['漲跌價差'].str.replace(
        'X', '').astype(dtype=float, errors='ignore')
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    indexNames = df1[df1['成交金額'] == 0].index
    df1 = df1.drop(indexNames)
    df1 = df1.reset_index(drop=True)
    df1.to_json('./stock_data/' + stockcode + '.json')
    return (df1)



def image_generate(DATE):  # 輸入時間、已存在json的股票代碼 輸出加上今日往前推算19日的指標圖片 日期須>=20100226
    pass


def get_list_ans(stockcode):

    PATH_input_json = './stock_data/'+stockcode+'.json'
    df1 = pd.read_json(PATH_input_json)

    for index, row in df1.iteritems():
        indexNames = df1[df1[index] == '--'].index
        df1 = df1.drop(indexNames)
        
    df1.sort_index(inplace=True)
    df1 = df1.fillna(0)
    df1 = df1.drop(range(14))
    df1 = df1.reset_index(drop=True)

    List_Low = np.array(df1['最低價'])
    List_Hig = np.array(df1['最高價'])
    List_price = np.array(df1['收盤價'])
    List_ans = np.zeros(19,dtype=int)
    List_ans_V3=np.zeros(19,dtype=int)
    for i in range(19, List_price.size-1):
        LL = float(List_Low[i-1])
        LH = float(List_Hig[i-1])
        CL = float(List_Low[i])
        CH = float(List_Hig[i])
        RL = float(List_Low[i+1])
        RH = float(List_Hig[i+1])
        LP = float(List_price[i-1])
        CP = float(List_price[i])
        RP = float(List_price[i + 1])

        cnt = 2
        while(CP == LP):
            LP = float(List_price[i-cnt])
            cnt = cnt+1
        cnt = 2
        while(CP == RP):
            RP = float(List_price[i+cnt])
            cnt = cnt + 1
        
        if LP > CP and RP>CP:
            List_ans = np.append(List_ans, int(1))
        elif CP > LP and CP>RP:
            List_ans = np.append(List_ans, int(-1))
        else:
            List_ans = np.append(List_ans, int(0))
        
        if RP > CP:
            List_ans_V3 = np.append(List_ans_V3,int(1))
        else:
            List_ans_V3 = np.append(List_ans_V3,int(-1))
        '''
        if LL>CH and RL>CH :
            List_ans=np.append(List_ans,1)
        elif CL>=LH and CL>RH:
            List_ans=np.append(List_ans,-1)
        else:
            List_ans=np.append(List_ans,0)
        '''
    List_ans = np.append(List_ans, 0)
    List_ans_V3=np.append(List_ans_V3, 0)
    import sys
    np.set_printoptions(threshold = sys.maxsize)
    return List_ans, List_price, List_ans_V3
    
def get_list_ans(stockcode):

    PATH_input_json = './stock_data/'+stockcode+'.json'
    df1 = pd.read_json(PATH_input_json)

    for index, row in df1.iteritems():
        indexNames = df1[df1[index] == '--'].index
        df1 = df1.drop(indexNames)
        
    df1.sort_index(inplace=True)
    df1 = df1.fillna(0)
    df1 = df1.drop(range(14))
    df1 = df1.reset_index(drop=True)

    List_Low = np.array(df1['最低價'])
    List_Hig = np.array(df1['最高價'])
    List_price = np.array(df1['收盤價'])
    List_ans = np.zeros(19,dtype=int)
    List_ans_V3=np.zeros(19,dtype=int)
    for i in range(19, List_price.size-1):
        LL = float(List_Low[i-1])
        LH = float(List_Hig[i-1])
        CL = float(List_Low[i])
        CH = float(List_Hig[i])
        RL = float(List_Low[i+1])
        RH = float(List_Hig[i+1])
        LP = float(List_price[i-1])
        CP = float(List_price[i])
        RP = float(List_price[i + 1])

        cnt = 2
        while(CP == LP):
            LP = float(List_price[i-cnt])
            cnt = cnt+1
        cnt = 2
        while(CP == RP):
            RP = float(List_price[i+cnt])
            cnt = cnt + 1
        
        if LP > CP and RP>CP:
            List_ans = np.append(List_ans, int(1))
        elif CP > LP and CP>RP:
            List_ans = np.append(List_ans, int(-1))
        else:
            List_ans = np.append(List_ans, int(0))
        
        if RP > CP:
            List_ans_V3 = np.append(List_ans_V3,int(1))
        else:
            List_ans_V3 = np.append(List_ans_V3,int(-1))
        '''
        if LL>CH and RL>CH :
            List_ans=np.append(List_ans,1)
        elif CL>=LH and CL>RH:
            List_ans=np.append(List_ans,-1)
        else:
            List_ans=np.append(List_ans,0)
        '''
    List_ans = np.append(List_ans, 0)
    List_ans_V3=np.append(List_ans_V3, 0)
    import sys
    np.set_printoptions(threshold = sys.maxsize)
    return List_ans, List_price, List_ans_V3
    
def get_list_ans_V4(stockcode):

    PATH_input_json = './stock_data/'+stockcode+'.json'
    df1 = pd.read_json(PATH_input_json)

    for index, row in df1.iteritems():
        indexNames = df1[df1[index] == '--'].index
        df1 = df1.drop(indexNames)
        
    df1.sort_index(inplace=True)
    df1 = df1.fillna(0)
    df1 = df1.drop(range(14))
    df1 = df1.reset_index(drop=True)

    List_price = np.array(df1['收盤價']).astype(np.float64)
    List_ans = np.zeros(19,dtype=int)
    for i in range(19, List_price.size - 1):
        Begin=i-19
        End=i-1
        Mid=i-10
        Max=i-19
        Min=i-19
        maxv=List_price[Begin]
        minv=List_price[Begin]
        for j in range(Begin,End):
            num=List_price[j]
            if(num>maxv):
                maxv=num
                Max=j
            elif(num<minv):
                minv=num
                Min=j
        if Max==Mid:
            List_ans = np.append(List_ans, -1)
        elif Min==Mid:
            List_ans = np.append(List_ans, 1)
        else:
            List_ans = np.append(List_ans, 0)
    List_ans = np.append(List_ans, 0)
    import sys
    np.set_printoptions(threshold = sys.maxsize)
    return List_ans, List_price



if __name__ == "__main__":
    get_list_ans('0051')
    pass
