#撈股票
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
import csv
import time
import os
#plt.rcParams['font.family']='SimHei' # 顯示中文('SimHei' for MacOS)
from sklearn import preprocessing


def checkCodeInDir(stockcode):  #查詢是否存在，存在回傳1 不存在回傳0 並建立
    PATH_TO_STOCKDATA = './stock_data'
    target = stockcode + '.json'
    for root, dirs, files in os.walk(PATH_TO_STOCKDATA):
        if target in files:
            print('find data at : '+root)
            return 1
        pass
    print("Data is not be found ,crawling data from ''https://www.twse.com.tw/exchangeReport/STOCK_DAY?'',\n please wait ")
    get_stockdata(stockcode)
    return 0
    


def get_stockdata(stockcode): #接從網路抓股票代碼 並儲存在./stock_data/XXXXX.json 裡面

    date = "2010/01/01"
    inday = datetime.datetime.strptime(date, "%Y/%m/%d")
    ago = [(inday + relativedelta(months= +i)).strftime('%Y%m%d') for i in range(109)]
    month = ago[0]

    data = {
        'response':'json',
        'date': month,
        'stockNo':stockcode,
    }
    dres = requests.get('https://www.twse.com.tw/exchangeReport/STOCK_DAY?', params=data)
    result = json.loads(dres.text)
    df1 = pd.DataFrame(result['data'])
    df1.columns = result['fields']
    print(month)
    time.sleep(3)

    for month in ago[1:109] :
        data = {
            'response':'json',
            'date': month,
            'stockNo':stockcode,
        }
        dres = requests.get('https://www.twse.com.tw/exchangeReport/STOCK_DAY?', params=data)
        result = json.loads(dres.text)
        tmp = pd.DataFrame(result['data'])
        tmp.columns = result['fields']
        df1=df1.append(tmp).reset_index(drop=True)
        print(month)
        time.sleep(3)

    df1['開盤價']=df1['開盤價'].str.replace(',','').astype(dtype=float,errors='ignore')
    df1['最高價']=df1['最高價'].str.replace(',','').astype(dtype=float,errors='ignore')
    df1['最低價']=df1['最低價'].str.replace(',','').astype(dtype=float,errors='ignore')
    df1['收盤價']=df1['收盤價'].str.replace(',','').astype(dtype=float,errors='ignore')
    df1['成交股數']=df1['成交股數'].str.replace(',','').astype(dtype=float,errors='ignore')
    df1['成交金額']=df1['成交金額'].str.replace(',','').astype(dtype=float,errors='ignore')
    df1['成交筆數']=df1['成交筆數'].str.replace(',','').astype(dtype=float,errors='ignore')
    df1['漲跌價差']=df1['漲跌價差'].str.replace('+','')
    df1['漲跌價差']=df1['漲跌價差'].str.replace('X','').astype(dtype=float,errors='ignore')
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    indexNames = df1[ df1['成交金額'] == 0 ].index
    df1 = df1.drop(indexNames)
    df1 = df1.reset_index(drop=True)
    df1.to_json('./stock_data/'+stockcode+'.json')
    df1


if __name__ == "__main__":
    checkCodeInDir("0050")

