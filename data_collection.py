import requests
from dateutil.relativedelta import relativedelta
import re
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0'}

print('Scraping data, please wait...')


#A function which extracts the symbols and names of companies listed on the Yahoo Finance trending page. We sort by alphabetical order
#of the names for later display in the dashboard.
def extract_symbols():   
    temp=[] 
    symbols=[]
    names=[]
    data=requests.get('https://uk.finance.yahoo.com/trending-tickers',headers=headers).content
    soup2=BeautifulSoup(data,'html.parser')
    table=soup2.select('tbody')[0]

    for tr in table.select('tr'):
        col=tr.select('td')[0]
        symbol=col.find('a')['href'].split('/')[2]
        name=col.find('a')['title']
        temp.append([name,symbol])
    temp=sorted(temp)
    names=[var[0] for var in temp]
    symbols=[var[1] for var in temp]
    return names,symbols

#A function which creates urls from the symbols to scrape financial data.
def create_urls(symbols):
    urls=[]
    for symbol in symbols:
        url='https://uk.finance.yahoo.com/quote/'+symbol+'/history/?period1='+str(time.time()-5*366*24*60*60)+'&period2='+str(time.time())
        urls.append(url)
    return urls

#Creating dataframes from the generated URLs. Some entries like FTSE 100 don't have historical data to pull from, so if we
#fail to find a specific class related to the currency during scraping, we remove that from our scraping. 
def create_dataframes(urls,symbols,names):
    dfs=[]
    currencies=[]
    for url in tqdm(urls):
        i=urls.index(url)
        data=requests.get(url,headers=headers).content
        soup=BeautifulSoup(data,'html.parser')
        currency=soup.find('div',{'class':'currency yf-j5d1ld'})
        try:
            currencies.append(currency.find('span').text.split()[2])
        except:
            symbols.remove(symbols[i])
            names.remove(names[i])
            continue
        table = soup.select('table')[0]
        columns = []
        for th in table.select('th'):
            columns.append(th.text.strip())
        columns = list(map(lambda x: x.replace('Adj Close      Adjusted closing price adjusted for splits and dividend and/or capital gain distributions.', 'Adj. Close'), columns))
        data = []
        for tr in table.select('tr'):
            row = []
            for td in tr.select('td'):
                row.append(td.text)
            if len(row):
                data.append(row)
        df = pd.DataFrame(data, columns=columns)
        dfs.append(df)
    return dfs,currencies,symbols,names

#The scraped data has some duplicate data related to dividends and other unrelated data, so we drop these. Given the high correlation between
#all of the numerical variables, we just keep the adjusted closing price for predictions.
def tidy_dataframes(dfs):
    newdfs=[]
    for df in dfs:
        df=df.drop_duplicates(['Date'],keep='last')
        df=df[['Date','Adj. Close']]
        df[['Date']]=df[['Date']].astype('datetime64[ns]')
        df=df.sort_values(['Date'])
        df = df.replace(',','', regex=True)
        df[['Adj. Close']]=df[['Adj. Close']].astype(float)
        df[['Adj. Close']]=df[['Adj. Close']].fillna(df[['Adj. Close']].mean())
        df=df.reset_index(drop=True)
        newdfs.append(df)
    return newdfs

#Data is missing on the weekends, so we fill this in with a simple forward fill.
def fill_missing_dates(dfs):
    newdfs=[]
    for df in dfs:
        date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D').to_numpy()
        df2=pd.DataFrame(date_range,columns=['Date'])
        df=df.merge(df2,how='right',on='Date')
        df=df.fillna(method='ffill')
        newdfs.append(df)
    return newdfs

#Generate new dataframes to let user filter by number of years for training data.
def additional_dataframes(newdfs):
    i=0
    for df in newdfs:
        symbol='\\'+symbols[i]+' ({})'.format(currencies[i])+'\\'
        latest_date=df['Date'].max()
        df_1Y=df[df['Date']>=latest_date-relativedelta(years=1)]
        df_3Y=df[df['Date']>=latest_date-relativedelta(years=3)]
        directory=os.getcwd()
        if not os.path.exists(directory+'\\stock_data\\'+symbol):
            os.makedirs(directory+'\\stock_data\\'+symbol)
        df_1Y.to_csv(directory+'\\stock_data\\'+symbol+'1Y',index=False)
        df_3Y.to_csv(directory+'\\stock_data\\'+symbol+'3Y',index=False)
        df.to_csv(directory+'\\stock_data\\'+symbol+'5Y',index=False)
        i+=1

#Web scrape the required data.
if __name__ == '__main__':
    names,symbols=extract_symbols()
    urls=create_urls(symbols)
    dfs,currencies,symbols,names=create_dataframes(urls,symbols,names)
    newdfs=fill_missing_dates(tidy_dataframes(dfs))
    additional_dataframes(newdfs)
    print('Stock data loaded!')