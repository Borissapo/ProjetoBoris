# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)

data0 = pd.read_csv('data.csv')

dataTest0 = pd.read_csv("test.csv")
dataTest0 = dataTest0.iloc[1:]  
data_complete = data0.append(dataTest0, ignore_index = True)

display(data_complete)

def sma(period,df):
    med=df[:period]
    for i in range(period,len(df)+1):
        value = round(np.mean(df[i-period:i]),3)
        med.append(value)
    return med

def info_compt(dados, name):
    sma3 = sma(3,list(dados[name]))
    sma15 = sma(15,list(dados[name]))
    sma30 = sma(30,list(dados[name]))
    sma45 = sma(45,list(dados[name]))
    sma60 = sma(60,list(dados[name]))
    sma90 = sma(90,list(dados[name]))
    sma120 = sma(120,list(dados[name]))
    mtx=[]
    for i in range(len(dados)):
        mtx.append([dados['DT_COMPTC'][i],sma3[i],sma15[i], sma30[i], sma45[i], sma60[i], sma90[i], sma120[i]])
    df =pd.DataFrame(mtx, columns=['DT_COMPTC',(name+"_sma3"),(name+'_sma15'),(name+'_sma30'), (name+'_sma45'), (name+'_sma60'), (name+'_sma90'), (name+'_sma120')])
    return df

df = info_compt(data_complete,'IBOV')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'Dol')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'NDX')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'SPX')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'DJI')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'selic')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'selic_cumulated')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'ipca')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'ipca_cumulated')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'igpm')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'igpm_cumulated')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'cdi')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'cdi_cumulated')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'Shanghai')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'EMBI')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'IBOV_FUT')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'IBrX_100')
data_complete = data_complete.merge(df, on='DT_COMPTC')
df = info_compt(data_complete,'IDol_FUT')
data_complete = data_complete.merge(df, on='DT_COMPTC')

display(data_complete)


inf = data_complete['ipca_cumulated'].to_numpy()
fluxo = data_complete['Fluxo'].to_numpy()
captc =  data_complete['CAPTC_DIA'].to_numpy()
resg =  data_complete['RESG_DIA'].to_numpy()
date = data_complete['DT_COMPTC'].to_numpy()

fluxo_fixed_ipca = (fluxo/(1+inf/100))
captc_fixed_ipca =  (captc/(1+inf/100))
resg_fixed_ipca =  (resg/(1+inf/100))

df = pd.DataFrame({"DT_COMPTC": date,'Fluxo_fixed_inf' : fluxo_fixed_ipca,'CAPTC_fixed_inf' : captc_fixed_ipca,'RESG_fixed_inf' : resg_fixed_ipca})

data_complete = data_complete.merge(df, on='DT_COMPTC', left_index=True)

display(data_complete)

data_final = data_complete.iloc[:-299]
dataTest_final = data_complete.iloc[-300:]

cor = data_final.corr()
display(cor[['RESG_fixed_inf']].nlargest(1000,'RESG_fixed_inf'))


data_final = data_final.drop(data_final[data_final["Fluxo"]>800].index)
data_final = data_final.drop(data_final[data_final["Fluxo"]<-800].index)

info0=['DT_COMPTC','IBOV_sma45','Dol','NDX_sma3','SPX_sma45','DJI_sma30','igpm_cumulated','selic_sma90']
info = info0+ ['Fluxo','CAPTC_fixed_inf','RESG_fixed_inf']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import  mean_squared_error

data = data_final

data = data[info]

wd =pd.to_datetime(data["DT_COMPTC"]).dt.weekday
wd = pd.get_dummies(wd)
data = wd.merge(data, left_index=True, right_index=True)
del(data["DT_COMPTC"])

npdata = data.to_numpy()

trainData, testData = train_test_split(npdata, test_size =0.3)

indice_final = data.shape[1]-3

indicec = data.shape[1]-2
indicer= data.shape[1]-1

X_train = trainData[:, :indice_final]
Y_train = trainData[:, indicec]

X_test = testData[:, :indice_final]
Y_test = testData[:,indicec]


regressorC = LinearRegression()
regressorC.fit(X_train, Y_train)

Q_testCap = regressorC.predict(X_test)

mse = mean_squared_error(Y_test, Q_testCap)
rmse = np.sqrt(mse)
print('Erro da captação no modelo de treino:',rmse)

X_train = trainData[:, :indice_final]
Y_train = trainData[:, indicer]

X_test = testData[:, :indice_final]
Y_test = testData[:,indicer]


regressorR = LinearRegression()
regressorR.fit(X_train, Y_train)

Q_testResg = regressorR.predict(X_test)

mse = mean_squared_error(Y_test, Q_testResg)
rmse = np.sqrt(mse)
print('Erro do resgate no modelo de treino:',rmse)

Q_test = Q_testCap - Q_testResg
Y_test = testData[:, indice_final]

mse = mean_squared_error(Y_test, Q_test)
rmse = np.sqrt(mse)
print('Erro do fluxo no modelo de treino:',rmse)

dataTest = dataTest_final

inf = dataTest['ipca_cumulated'].to_numpy()

dataTest = dataTest[info0]

wdTest =pd.to_datetime(dataTest["DT_COMPTC"]).dt.weekday
wdTest = pd.get_dummies(wdTest)
dataTest = wdTest.merge(dataTest, left_index=True,right_index=True)

time = pd.DataFrame(dataTest["DT_COMPTC"], columns =['DT_COMPTC'])
del(dataTest["DT_COMPTC"])

npTest = dataTest.to_numpy()

Q_C = regressorC.predict(npTest[:,:])
Q_R = regressorR.predict(npTest[:,:])

Q_Cinf = Q_C*(1 + 0.01*inf)
Q_Rinf = Q_R*(1 + 0.01*inf)

Q_flux = Q_Cinf-Q_Rinf

df_teste = pd.DataFrame({"DT_COMPTC": time["DT_COMPTC"].to_numpy(),'Fluxo' : Q_flux})

fluxo_real = (pd.read_csv('fluxo_real.csv'))

fluxo_r = (fluxo_real[['Fluxo']].to_numpy())/1000000

pos=0
neg=0
for i in range(len(df_teste)):
    if df_teste['Fluxo'][i]>0 and fluxo_r[i]>0:
        pos+=1
    if df_teste['Fluxo'][i]<0 and fluxo_r[i]<0:
        neg+=1
print('pos',pos)
print('neg',neg)
print('porcent de acerto do sentido:',100*(pos+neg)/len(df_teste))

mse = mean_squared_error(fluxo_r, df_teste['Fluxo'])
rmse = np.sqrt(mse)
print('Erro real:',rmse)

plt.plot(fluxo_r, color = 'red')
plt.plot(df_teste['Fluxo'], color = 'blue')

plt.show()

df_teste.to_csv('submission.csv', index = False)

