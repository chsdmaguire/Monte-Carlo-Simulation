from distutils.log import error
from operator import index
from statistics import mean
from matplotlib.pyplot import axis
import pandas as pd
from sklearn.metrics import mean_squared_error
import sqlalchemy
import requests
import datetime
import numpy as np

indicesDf = pd.read_csv('indices.csv')
msciDf = pd.read_csv('HistoricalPrices.csv', names=['Date', 'Open', 'High', 'Low', 'Close'])
msciDf2 = msciDf.drop(columns=['Open', 'High', 'Low'], axis=0)
msciDf2['series_id'] = 'MSCI'
msciDf2 = msciDf2.iloc[1:, :]
msciDf2 = msciDf2.rename(columns={'Date': 'time', 'Close': 'value'})
msciDf3 = msciDf2[['series_id', 'time', 'value']]
msciDf3['time'] = pd.to_datetime(msciDf3['time'])
indicesDf['time'] = pd.to_datetime(indicesDf['time'])

concatDf = pd.concat([indicesDf, msciDf3])
concatDf = concatDf.reset_index()
concatDf = concatDf.drop('index', axis=1)

pivDf = concatDf.pivot(index='time', columns='series_id', values='value')
pivDf = pivDf[pivDf['WILLLRGCAPGR'].notna()]

cols = pivDf.columns
pivDf[cols] = pivDf[cols].apply(pd.to_numeric, errors='coerce', axis=1)
pctDf = pivDf.pct_change()
pctDf = pctDf[pctDf['WILLLRGCAPGR'].notna()]

meanDf = pd.DataFrame()
stdDf = pd.DataFrame()
for (name, data) in pctDf.iteritems():
    d = {'series_id': [name], 'mean': [data.mean()]}
    newDf = pd.DataFrame(d)
    meanDf = meanDf.append(newDf)

    s = {'series_id': [name], 'std': [data.std()]}
    sDf = pd.DataFrame(s)
    stdDf = stdDf.append(sDf)

stdDf['annual_std'] = stdDf['std'] * (252 ** (1/2))

meanDf['annual_return'] = ((meanDf['mean'] + 1) ** 252) -1


yieldsDf = pd.read_csv('yields.csv', names=['series_id', 'time', 'value'])
yieldsDf = yieldsDf.iloc[1:, :]
yieldsDf['value'] = pd.to_numeric(yieldsDf['value'])
yieldsDf['value'] = yieldsDf['value'] / 100
yieldsDf['time'] = pd.to_datetime(yieldsDf['time'])
yieldsPiv = yieldsDf.pivot(index='time', columns='series_id', values='value')
cols = yieldsPiv.columns
yieldsPiv[cols] = yieldsPiv[cols].apply(pd.to_numeric, errors='coerce', axis=1)

#-----------------------------------------------
# Covariance matrix
covDf = yieldsPiv.merge(pctDf, on='time')
COV = covDf.cov()
#-----------------------------------------------
for (name, data) in yieldsPiv.iteritems():
    d = {'series_id': [name], 'mean': [np.nan], 'annual_return': [data.mean()]}
    newDf = pd.DataFrame(d)
    meanDf = meanDf.append(newDf)

    s = {'series_id': [name], 'std': [np.nan], 'annual_std': [data.std()]}
    sDf = pd.DataFrame(s)
    stdDf = stdDf.append(sDf)

mergeAnnuals = meanDf.merge(stdDf, on='series_id')
mergeAnnuals = mergeAnnuals.drop(columns=['mean', 'std'])


def weightscreator(df):
    df['weights'] = np.random.random(len(df))
    df['weights'] /= df['weights'].sum()
    return df

def portfoliostd(df, weights):
    var = df['annual_std'] * 100

    shape =  np.reshape(np.array(weights ).T, (16, 1))
    frst = np.dot(var, weights)
    prod = np.dot(np.reshape(np.dot(COV, shape), (1, 16)), weights *2)
    return (frst + prod) / 100

def portfolioreturn(df, weights):
    return np.dot(df['annual_return'], weights)

returns = []
stds = []
w = []

prtfoliosDf = pd.DataFrame()

for i in range(50000):
    weights = weightscreator(mergeAnnuals)
    weights2 = weights[['series_id', 'weights']]
    pivWeights = weights2.pivot_table(columns='series_id', values='weights')
    pivWeights['return'] = portfolioreturn(mergeAnnuals, weights['weights'])
    pivWeights['std'] = portfoliostd(mergeAnnuals, weights['weights'])
    print(pivWeights)
    prtfoliosDf = prtfoliosDf.append(pivWeights)

prtfoliosDf.to_csv('markovData.csv')