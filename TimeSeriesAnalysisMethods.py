#!/usr/bin/env python
# coding: utf-8

# # NBA POPULARITY updated

# In[1]:


#Author: Mert Calis
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[2]:


rawUsa = pd.read_csv('NbaUsa.csv',skiprows=1,parse_dates=True)
rawPl = pd.read_csv('NbaPoland.csv',skiprows=1,parse_dates=True)
rawTr = pd.read_csv('NbaTurkey.csv',skiprows=1,parse_dates=True)


# In[ ]:





# In[3]:


rawUsa.head()
rawUsa.rename(columns={"nba: (Amerika Birleşik Devletleri)": "USA"},inplace=True)
rawPl.rename(columns={"nba: (Polonya)":"PL"},inplace=True)
rawTr.rename(columns={"nba: (Türkiye)":"TR"},inplace=True)


# In[4]:


df = rawPl.merge(rawUsa, on="Ay").merge(rawTr, on="Ay")
df["Date"] = df["Ay"].astype(str)
df["Date"]=pd.to_datetime(df["Date"])
df.set_index('Date',inplace=True)
df.index.freq ='MS'
df.drop('Ay',inplace=True,axis=1)
df.head()


# In[5]:


df.plot(figsize=(15,8))


# In[6]:


df.isna().sum()


# In[7]:


auto_arima(df['USA'],seasonal=True,m=12,trace=True)


# In[ ]:





# # Seasonality, Stationary Check

# In[8]:


decompose_dataPL = seasonal_decompose(df['PL'],model="Additive")
decompose_dataPL.plot();


# In[9]:


seasonalityPL = decompose_dataPL.seasonal
seasonalityPL.plot(figsize=(15,8),color='green')


# In[10]:


decompose_dataUSA = seasonal_decompose(df['USA'],model="Additive")
decompose_dataUSA.plot();


# In[11]:


seasonalityUSA = decompose_dataUSA.seasonal
seasonalityUSA.plot(figsize=(15,8),color='green')


# In[12]:


decompose_dataTR = seasonal_decompose(df['TR'],model="Additive")
decompose_dataTR.plot();


# In[13]:


seasonalityTR = decompose_dataTR.seasonal
seasonalityTR.plot(figsize=(15,8),color='green')


# As we can clearly see at first glance both data are seasonal
# We need stationary data for some forecasting methods
# That's why I will use rolling mean differentiation

# In[14]:


diffPL = df['PL'].rolling(window = 12).mean()
diffUSA = df['USA'].rolling(window = 12).mean()
diffTR = df['TR'].rolling(window = 12).mean()


# In[15]:


diffPL.plot()
diffUSA.plot()
diffTR.plot()


# # adfuller test

# In[16]:


dfadfTestUSA = adfuller(diffUSA.dropna(), autolag = 'AIC')
print("--USA--")
print("1. ADF : ",dfadfTestUSA[0])
print("2. P-Value : ", dfadfTestUSA[1])
print("3. Num Of Lags : ", dfadfTestUSA[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dfadfTestUSA[3])
print("5. Critical Values :")
for key, val in dfadfTestUSA[4].items():
    print("\t",key, ": ", val)

dfadfTestPL = adfuller(diffPL.dropna(), autolag = 'AIC')
print("--PL--")
print("1. ADF : ",dfadfTestPL[0])
print("2. P-Value : ", dfadfTestPL[1])
print("3. Num Of Lags : ", dfadfTestPL[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dfadfTestPL[3])
print("5. Critical Values :")
for key, val in dfadfTestPL[4].items():
    print("\t",key, ": ", val)

dfadfTestTR = adfuller(diffTR.dropna(), autolag = 'AIC')
print("--TR--")
print("1. ADF : ",dfadfTestTR[0])
print("2. P-Value : ", dfadfTestTR[1])
print("3. Num Of Lags : ", dfadfTestTR[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dfadfTestTR[3])
print("5. Critical Values :")
for key, val in dfadfTestTR[4].items():
    print("\t",key, ": ", val)


# --For both of them we can see P > 0.05 So Time Series is Non-Stationary---

# # Preparing Data for Training

# In[17]:


ncut = len(df)-24
df_train = df.iloc[:ncut]
df_test = df.iloc[ncut:]
start = len(df_train)
stop = len(df)-1

df_diffUSA_train = diffUSA.iloc[:ncut]
df_diffUSA_test = diffUSA.iloc[ncut:]

plt.title("USA")
df_train['USA'].plot(figsize=(14,5),legend = True, label='Train')
df_test['USA'].plot(figsize=(14,5),legend = True, label='Test')


# In[18]:


modelSARIMAX_USA =SARIMAX(df_train['USA'],order=(1, 1, 3), seasonal_order=(1, 0, 1, 12))
resultSARIMAX_USA = modelSARIMAX_USA.fit()
fcastSARIMAX_USA  = resultSARIMAX_USA.predict(start = start, end = stop, dynamic = False, typ = 'levels').rename('SARIMAX')


# # Implement this method for SARIMAX

# In[19]:


plt.figure(figsize=(15,8))
fcastSARIMAX_USA.plot(legend =True)
df_test['USA'].plot(legend =True)


# In[21]:


print('MSE:',mean_squared_error(fcastSARIMAX_USA,df_test['USA']))
print('Mean Absolute Error: ',mean_absolute_error(fcastSARIMAX_USA,df_test['USA']))


# In[24]:


auto_arima(df['PL'],seasonal=True,m=12,trace=True)


# In[25]:


modelSARIMAX_PL =SARIMAX(df_train['PL'],order=(0, 1, 3), seasonal_order=(1, 0, 1, 12))
resultSARIMAX_PL = modelSARIMAX_PL.fit()
fcastSARIMAX_PL  = resultSARIMAX_PL.predict(start = start, end = stop, dynamic = False, typ = 'levels').rename('SARIMAX')

plt.figure(figsize=(15,8))
fcastSARIMAX_PL.plot(legend =True)
df_test['PL'].plot(legend =True)


# In[26]:


print('MSE:',mean_squared_error(fcastSARIMAX_PL,df_test['PL']))
print('Mean Absolute Error: ',mean_absolute_error(fcastSARIMAX_PL,df_test['PL']))


# In[28]:


auto_arima(df['TR'],seasonal=True,m=12,trace=True)


# In[30]:


modelSARIMAX_TR =SARIMAX(df_train['TR'],order=(2, 1, 1), seasonal_order=(2, 0, 1, 12))
resultSARIMAX_TR = modelSARIMAX_TR.fit()
fcastSARIMAX_TR  = resultSARIMAX_TR.predict(start = start, end = stop, dynamic = False, typ = 'levels').rename('SARIMAX')

plt.figure(figsize=(15,8))
fcastSARIMAX_TR.plot(legend =True)
df_test['TR'].plot(legend =True)


# In[31]:


print('MSE:',mean_squared_error(fcastSARIMAX_TR,df_test['TR']))
print('Mean Absolute Error: ',mean_absolute_error(fcastSARIMAX_TR,df_test['TR']))


# In[65]:


plt.title("PL")
df_train['PL'].plot(figsize=(14,5),legend = True, label='Train')
df_test['PL'].plot(figsize=(14,5),legend = True, label='Test')


# In[66]:


plt.title("TR")
df_train['TR'].plot(figsize=(14,5),legend = True, label='Train')
df_test['TR'].plot(figsize=(14,5),legend = True, label='Test')


# # SES and HOLT'S METHODS

# In[32]:


from statsmodels.tsa.api import (SimpleExpSmoothing, Holt, ExponentialSmoothing)


# In[33]:


fitSES_USA = SimpleExpSmoothing(df_train['USA'].to_numpy()).fit()
fcastSES_USA = fitSES_USA.forecast(len(df_test['USA']))

fitHOLT_USA = Holt(df_train['USA'].to_numpy(),exponential=False).fit()
fcastHOLT_USA = fitHOLT_USA.forecast(len(df_test['USA']))

fitSES_PL = SimpleExpSmoothing(df_train['PL'].to_numpy()).fit()
fcastSES_PL = fitSES_PL.forecast(len(df_test['PL']))

fitHOLT_PL = Holt(df_train['PL'].to_numpy(),exponential=False).fit()
fcastHOLT_PL = fitHOLT_PL.forecast(len(df_test['PL']))

fitSES_TR = SimpleExpSmoothing(df_train['TR'].to_numpy()).fit()
fcastSES_TR = fitSES_TR.forecast(len(df_test['TR']))

fitHOLT_TR = Holt(df_train['TR'].to_numpy(),exponential=False).fit()
fcastHOLT_TR = fitHOLT_TR.forecast(len(df_test['TR']))


# In[34]:


fitSES_diffUSA = SimpleExpSmoothing(df_diffUSA_train.to_numpy()).fit()
fcastSES_diffUSA = fitSES_diffUSA.forecast(len(df_diffUSA_test))

fitHOLT_diffUSA = Holt(df_diffUSA_train.to_numpy(),exponential=False).fit()
fcastHOLT_diffUSA = fitHOLT_USA.forecast(len(df_diffUSA_test))


# In[35]:


df_SES_USA = pd.DataFrame(fcastSES_USA)
df_SES_USA["Date"]=df_test.index
df_SES_USA.set_index('Date',inplace=True)
df_SES_USA.rename(columns={0:'USA'},inplace = True)

df_Holt_USA = pd.DataFrame(fcastHOLT_USA)
df_Holt_USA["Date"]= df_test.index
df_Holt_USA.set_index('Date',inplace=True)
df_Holt_USA.rename(columns={0:'USA'},inplace = True)

df_SES_PL = pd.DataFrame(fcastSES_PL)
df_SES_PL["Date"]=df_test.index
df_SES_PL.set_index('Date',inplace=True)
df_SES_PL.rename(columns={0:'PL'},inplace = True)

df_Holt_PL = pd.DataFrame(fcastHOLT_PL)
df_Holt_PL["Date"]= df_test.index
df_Holt_PL.set_index('Date',inplace=True)
df_Holt_PL.rename(columns={0:'PL'},inplace = True)

df_SES_TR = pd.DataFrame(fcastSES_TR)
df_SES_TR["Date"]=df_test.index
df_SES_TR.set_index('Date',inplace=True)
df_SES_TR.rename(columns={0:'TR'},inplace = True)

df_Holt_TR = pd.DataFrame(fcastHOLT_TR)
df_Holt_TR["Date"]= df_test.index
df_Holt_TR.set_index('Date',inplace=True)
df_Holt_TR.rename(columns={0:'TR'},inplace = True)

df_SES_USAdiff = pd.DataFrame(fcastSES_diffUSA)
df_SES_USAdiff["Date"]=df_test.index
df_SES_USAdiff.set_index('Date',inplace=True)
df_SES_USAdiff.rename(columns={0:'USA'},inplace = True)

df_Holt_USAdiff = pd.DataFrame(fcastHOLT_diffUSA)
df_Holt_USAdiff["Date"]= df_test.index
df_Holt_USAdiff.set_index('Date',inplace=True)
df_Holt_USAdiff.rename(columns={0:'USA'},inplace = True)


# In[36]:


plt.figure()
plt.title('United States of America')
df_test['USA'].plot(figsize = (14,5),legend=True,label='Original Data')
df_SES_USA['USA'].plot(figsize = (14,5),legend=True,label='Simple Exponential Smoothing')
df_Holt_USA['USA'].plot(figsize = (14,5),legend=True,label='Holts Method')


# In[37]:


plt.figure()
plt.title('United States of America')
df_test['USA'].plot(figsize = (14,5),legend=True,label='Original Data')
df_SES_USAdiff['USA'].plot(figsize = (14,5),legend=True,label='Simple Exponential Smoothing(Differenciated)')
df_Holt_USAdiff['USA'].plot(figsize = (14,5),legend=True,label='Holts Method(Differenciated)')


# In[73]:


plt.figure()
plt.title('POLAND')
df_test['PL'].plot(figsize = (14,5),legend=True,label='Original Data')
df_SES_PL['PL'].plot(figsize = (14,5),legend=True,label='Simple Exponential Smoothing')
df_Holt_PL['PL'].plot(figsize = (14,5),legend=True,label='Holts Method')


# In[74]:


plt.figure()
plt.title('Turkey')
df_test['TR'].plot(figsize = (14,5),legend=True,label='Original Data')
df_SES_TR['TR'].plot(figsize = (14,5),legend=True,label='Simple Exponential Smoothing')
df_Holt_TR['TR'].plot(figsize = (14,5),legend=True,label='Holts Method')


# # Holt's Winter Method 

# In[41]:


fitHW1_USA = ExponentialSmoothing(df_train['USA'],seasonal_periods=12,trend="add",seasonal="add").fit()
fcastHW1_USA= fitHW1_USA.forecast(df_test['USA'].size)

fitHW2_USA = ExponentialSmoothing(df_train['USA'],seasonal_periods=12,trend="add",seasonal="mul").fit()
fcastHW2_USA= fitHW2_USA.forecast(df_test['USA'].size)

fitHW3_USA = ExponentialSmoothing(df_train['USA'],seasonal_periods=12,trend="mul",seasonal="add").fit()
fcastHW3_USA = fitHW3_USA.forecast(df_test['USA'].size)

fitHW4_USA = ExponentialSmoothing(df_train['USA'],seasonal_periods=12,trend="mul",seasonal="mul").fit()
fcastHW4_USA= fitHW4_USA.forecast(df_test['USA'].size)


# In[42]:


plt.figure()
plt.title("United States of America")
ax = df_test['USA'].plot(figsize=(14,5),color ='black',label='Test',legend=True)
fcastHW1_USA.rename("Trend:Add, Seasonal:Add").plot(ax=ax,color='red',legend=True)
fcastHW2_USA.rename("Trend:Add, Seasonal:Mul").plot(ax=ax,color='green',legend=True)
fcastHW3_USA.rename("Trend:Mul, Seasonal:Add").plot(ax=ax,color='blue',legend=True)
fcastHW4_USA.rename("Trend:Mul, Seasonal:Mul").plot(ax=ax,color='gray',legend=True)
plt.show()


# In[43]:


print('MSE add, add:',mean_squared_error(df_test['USA'],fcastHW1_USA))
print('MSE add, mul:',mean_squared_error(df_test['USA'],fcastHW2_USA))
print('MSE mul, add:',mean_squared_error(df_test['USA'],fcastHW3_USA))
print('MSE mul, mul:',mean_squared_error(df_test['USA'],fcastHW4_USA))


# (add,mul) and (add,add) looks similar but little -40 shifted

# In[44]:


fitHW1_PL = ExponentialSmoothing(df_train['PL'],seasonal_periods=12,trend="add",seasonal="add").fit()
fcastHW1_PL= fitHW1_PL.forecast(df_test['PL'].size)

fitHW2_PL = ExponentialSmoothing(df_train['PL'],seasonal_periods=12,trend="add",seasonal="mul").fit()
fcastHW2_PL= fitHW2_PL.forecast(df_test['PL'].size)

fitHW3_PL = ExponentialSmoothing(df_train['PL'],seasonal_periods=12,trend="mul",seasonal="add").fit()
fcastHW3_PL = fitHW3_PL.forecast(df_test['PL'].size)

fitHW4_PL = ExponentialSmoothing(df_train['PL'],seasonal_periods=12,trend="mul",seasonal="mul").fit()
fcastHW4_PL= fitHW4_PL.forecast(df_test['PL'].size)


# In[45]:


print('MSE add, add:',mean_squared_error(df_test['PL'],fcastHW1_PL))
print('MSE add, mul:',mean_squared_error(df_test['PL'],fcastHW2_PL))
print('MSE mul, add:',mean_squared_error(df_test['PL'],fcastHW3_PL))
print('MSE mul, mul:',mean_squared_error(df_test['PL'],fcastHW4_PL))


# In[46]:


plt.figure()
plt.title("Poland")
ax = df_test['PL'].plot(figsize=(14,5),color ='black',label='Test',legend=True)
fcastHW1_PL.rename("Trend:Add, Seasonal:Add").plot(ax=ax,color='red',legend=True)
fcastHW2_PL.rename("Trend:Add, Seasonal:Mul").plot(ax=ax,color='green',legend=True)
fcastHW3_PL.rename("Trend:Mul, Seasonal:Add").plot(ax=ax,color='blue',legend=True)
fcastHW4_PL.rename("Trend:Mul, Seasonal:Mul").plot(ax=ax,color='gray',legend=True)
plt.show()


# In[47]:


fitHW1_TR = ExponentialSmoothing(df_train['TR'],seasonal_periods=12,trend="add",seasonal="add").fit()
fcastHW1_TR= fitHW1_TR.forecast(df_test['TR'].size)

fitHW2_TR = ExponentialSmoothing(df_train['TR'],seasonal_periods=12,trend="add",seasonal="mul").fit()
fcastHW2_TR= fitHW2_TR.forecast(df_test['TR'].size)

fitHW3_TR = ExponentialSmoothing(df_train['TR'],seasonal_periods=12,trend="mul",seasonal="add").fit()
fcastHW3_TR = fitHW3_TR.forecast(df_test['TR'].size)

fitHW4_TR = ExponentialSmoothing(df_train['TR'],seasonal_periods=12,trend="mul",seasonal="mul").fit()
fcastHW4_TR = fitHW4_TR.forecast(df_test['TR'].size)


# In[48]:


print('MSE add, add:',mean_squared_error(df_test['TR'],fcastHW1_TR))
print('MSE add, mul:',mean_squared_error(df_test['TR'],fcastHW2_TR))
print('MSE mul, add:',mean_squared_error(df_test['TR'],fcastHW3_TR))
print('MSE mul, mul:',mean_squared_error(df_test['TR'],fcastHW4_TR))


# In[49]:


plt.figure()
plt.title("Turkey")
ax = df_test['TR'].plot(figsize=(14,5),color ='black',label='Test',legend=True)
fcastHW1_TR.rename("Trend:Add, Seasonal:Add").plot(ax=ax,color='red',legend=True)
fcastHW2_TR.rename("Trend:Add, Seasonal:Mul").plot(ax=ax,color='green',legend=True)
fcastHW3_TR.rename("Trend:Mul, Seasonal:Add").plot(ax=ax,color='blue',legend=True)
fcastHW4_TR.rename("Trend:Mul, Seasonal:Mul").plot(ax=ax,color='gray',legend=True)
plt.show()


# # ARIMA

# In[50]:


from statsmodels.tsa.arima.model import ARIMA


# In[51]:


def ARIMAForecast(train,test,order):
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=order).fit()
        y = model.forecast()[0]
        predictions.append(y)
        history.append(test[t])
    return predictions


# In[52]:


ArimaVals_USA = ARIMAForecast(df_train['USA'],df_test['USA'],(0,1,3))
fcast_ARIMA_USA = pd.DataFrame(ArimaVals_USA)
fcast_ARIMA_USA["Date"] = df_test.index
fcast_ARIMA_USA.set_index('Date',inplace=True)
fcast_ARIMA_USA.rename(columns={0:'USA'},inplace=True)

plt.figure()
plt.title("United States of America")
df_test['USA'].plot(figsize=(14,5),legend=True,label='Original Data')
fcast_ARIMA_USA['USA'].plot(figsize=(14,5),legend=True,label='ARIMA')


# In[53]:


print('MSE :',mean_squared_error(df_test['USA'],fcast_ARIMA_USA['USA']))


# In[54]:


ArimaVals_PL = ARIMAForecast(df_train['PL'],df_test['PL'],(0,1,3))
fcast_ARIMA_PL = pd.DataFrame(ArimaVals_PL)
fcast_ARIMA_PL["Date"] = df_test.index
fcast_ARIMA_PL.set_index('Date',inplace=True)
fcast_ARIMA_PL.rename(columns={0:'PL'},inplace=True)

plt.figure()
plt.title("Poland")
df_test['PL'].plot(figsize=(14,5),legend=True,label='Original Data')
fcast_ARIMA_PL['PL'].plot(figsize=(14,5),legend=True,label='ARIMA')


# In[55]:


print('MSE :',mean_squared_error(df_test['PL'],fcast_ARIMA_PL['PL']))


# In[56]:


ArimaVals_TR = ARIMAForecast(df_train['TR'],df_test['TR'],(0,1,3))
fcast_ARIMA_TR = pd.DataFrame(ArimaVals_TR)
fcast_ARIMA_TR["Date"] = df_test.index
fcast_ARIMA_TR.set_index('Date',inplace=True)
fcast_ARIMA_TR.rename(columns={0:'TR'},inplace=True)

plt.figure()
plt.title("Turkey")
df_test['TR'].plot(figsize=(14,5),legend=True,label='Original Data')
fcast_ARIMA_TR['TR'].plot(figsize=(14,5),legend=True,label='ARIMA')


# In[57]:


print('MSE :',mean_squared_error(df_test['TR'],fcast_ARIMA_TR['TR']))


# # SARIMA

# In[36]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:





# In[37]:


def find_best_SARIMAX_values(train,test,pList,dList,q1List,q2List,q3List,q4List):
    mse = 10000000
    bestCoefficients = (0,0,(0,0,0,0))
    for p in pList:
        for d in dList:
            for q1 in q1List:
                for q2 in q2List:
                    for q3 in q3List:
                        for q4 in q4List:
                            fit1 = SARIMAX(train,order=(p,d,(q1,q2,q3,q4))).fit()
                            fcast = fit1.forecast(len(test))
                            meansq = mean_squared_error(fcast,test)
                            if(meansq<mse):
                                mse = meansq
                                bestCoefficients = (p,d,(q1,q2,q3,q4))
    
    return bestCoefficients


# In[38]:


diff_train = df_train.diff().dropna()
diff_test = df_test.diff().dropna()

p = [0,1,2,4,8]
d = [1]
q1=[0,1]
q2=[0,1]
q3=[0,1]
q4=[0,1]
best_fitted_coef1=find_best_SARIMAX_values(df_train['USA'],df_test['USA'],p,d,q1,q2,q3,q4)
best_fitted_coef2=find_best_SARIMAX_values(diff_train['PL'],diff_test['PL'],p,d,q1,q2,q3,q4)
best_fitted_coef3=find_best_SARIMAX_values(diff_train['TR'],diff_test['TR'],p,d,q1,q2,q3,q4)
print(best_fitted_coef1)
print(best_fitted_coef2)
print(best_fitted_coef3)


# In[39]:


fitSAR_USA = SARIMAX(diff_train['USA'],order=best_fitted_coef1).fit()
fcastSAR_USA = fitSAR_USA.forecast(len(diff_test),dynamic=True)
fcast_SARIMA_USA = pd.DataFrame(fcastSAR_USA)
fcast_SARIMA_USA["Date"] = diff_test.index
fcast_SARIMA_USA.set_index('Date',inplace=True)
fcast_SARIMA_USA.rename(columns={'predicted_mean':'USA'},inplace=True)

diff_test['USA'].plot(figsize=(14,5),legend=True,label='Original Data')
fcast_SARIMA_USA['USA'].plot(figsize=(14,5),legend=True,label='SARIMA')


# In[41]:


fitSAR_PL = SARIMAX(diff_train['PL'],order=best_fitted_coef2).fit()
fcastSAR_PL = fitSAR_PL.forecast(len(diff_test),dynamic=True)
fcast_SARIMA_PL = pd.DataFrame(fcastSAR_PL)
fcast_SARIMA_PL["Date"] = diff_test.index
fcast_SARIMA_PL.set_index('Date',inplace=True)
fcast_SARIMA_PL.rename(columns={'predicted_mean':'PL'},inplace=True)

diff_test['PL'].plot(figsize=(14,5),legend=True,label='Original Data')
fcast_SARIMA_PL['PL'].plot(figsize=(14,5),legend=True,label='SARIMA')


# In[42]:


fitSAR_TR = SARIMAX(diff_train['TR'],order=best_fitted_coef3).fit()
fcastSAR_TR = fitSAR_TR.forecast(len(diff_test),dynamic=True)
fcast_SARIMA_TR = pd.DataFrame(fcastSAR_TR)
fcast_SARIMA_TR["Date"] = diff_test.index
fcast_SARIMA_TR.set_index('Date',inplace=True)
fcast_SARIMA_TR.rename(columns={'predicted_mean':'TR'},inplace=True)

diff_test['TR'].plot(figsize=(14,5),legend=True,label='Original Data')
fcast_SARIMA_TR['TR'].plot(figsize=(14,5),legend=True,label='SARIMA')


# # Random Forest

# In[77]:


from sklearn.ensemble import RandomForestRegressor


# In[236]:


df.head()


# In[230]:


df1_USA = df.diff().dropna()
df1_PL = df.diff().dropna()
df1_TR = df.diff().dropna()


# In[106]:


df1_USA = df.copy()
df1_PL = df.copy()
df1_TR = df.copy()


# In[231]:


df1_TR


# In[232]:


df1_USA.drop('PL',inplace=True,axis=1)
df1_USA.drop('TR',inplace=True,axis=1)

df1_PL.drop('USA',inplace=True,axis=1)
df1_PL.drop('TR',inplace=True,axis=1)

df1_TR.drop('USA',inplace=True,axis=1)
df1_TR.drop('PL',inplace=True,axis=1)


# In[ ]:





# In[233]:


for i in range(12,0,-1):
    df1_USA['t- ' + str(i)]=df1_USA['USA'].shift(i)
    
for i in range(12,0,-1):
    df1_PL['t- ' + str(i)]=df1_PL['PL'].shift(i)

for i in range(12,0,-1):
    df1_TR['t- ' + str(i)]=df1_TR['TR'].shift(i)


# In[234]:


df1_USA.dropna(inplace=True)
df1_PL.dropna(inplace=True)
df1_TR.dropna(inplace=True)


# In[235]:


df1_PL


# In[244]:


X_USA = df1_USA.iloc[:,1:].values
Y_USA = df1_USA.iloc[:,0].values

X_PL = df1_PL.iloc[:,1:].values
Y_PL = df1_PL.iloc[:,0].values

X_TR = df1_TR.iloc[:,1:].values
Y_TR = df1_TR.iloc[:,0].values


# In[246]:


df1_PL.iloc[:,1:]


# In[245]:


X_PL


# In[247]:


Y_PL


# In[248]:


nSplit = int(len(X_USA)*0.80)
nSplit


# In[249]:


X_USA_train, X_USA_test = X_USA[0:nSplit], X_USA[nSplit:]
Y_USA_train, Y_USA_test = Y_USA[0:nSplit], Y_USA[nSplit:]

X_PL_train, X_PL_test = X_PL[0:nSplit], X_PL[nSplit:]
Y_PL_train, Y_PL_test = Y_PL[0:nSplit], Y_PL[nSplit:]

X_TR_train, X_TR_test = X_TR[0:nSplit], X_TR[nSplit:]
Y_TR_train, Y_TR_test = Y_TR[0:nSplit], Y_TR[nSplit:]


# In[251]:


rfr_USA=RandomForestRegressor(n_estimators=200)
rfr_USA.fit(X_USA_train,Y_USA_train)


# In[258]:


rfr_PL=RandomForestRegressor(n_estimators=200)
rfr_PL.fit(X_PL_train,Y_PL_train)
print(rfr_PL.score(X_PL_train,Y_PL_train))
print(rfr_PL.score(X_PL_test,Y_PL_test))


# In[259]:


train_prediction=rfr_PL.predict(X_PL_train)
test_prediction=rfr_PL.predict(X_PL_test)
plt.scatter(test_prediction,Y_PL_test);


# In[135]:


from sklearn.model_selection import ParameterGrid


# In[388]:


grid ={'n_estimators': range(20,200,10),'max_depth': [2,3,4,5,6,7,8,9,10],'max_features':[2,3,4,5,6,7,8,9], 'random_state' : [18,19]}


# In[389]:


rfr = RandomForestRegressor()
test_scores = []

for g in ParameterGrid(grid):
    rfr.set_params(**g)
    rfr.fit(X_PL_train,Y_PL_train)
    test_scores.append(rfr.score(X_PL_test,Y_PL_test))


# In[261]:


best_idx = np.argmax(test_scores)
print(test_scores[best_idx],ParameterGrid(grid)[best_idx])


# In[386]:


rfr = RandomForestRegressor(n_estimators=20,max_features=2,max_depth=5)
rfr.fit(X_PL_train,Y_PL_train)
print(rfr.score(X_PL_train,Y_PL_train))
print(rfr.score(X_PL_test,Y_PL_test))


# In[61]:


feature_importanceRFR = rfr.feature_importances_


# In[63]:


sorted_index = np.argsort(feature_importanceRFR)[::1]
x1=range(len(feature_importanceRFR))
df1_USA.columns.to_list()


# In[64]:


feature_names=df1_USA.columns.tolist()[1:]
labels = np.array(feature_names)[sorted_index]


# In[65]:


plt.bar(x1,feature_importanceRFR[sorted_index],tick_label=labels)
plt.xticks(rotation=90);


# # Gradient Boosted Trees

# In[66]:


from sklearn.ensemble import GradientBoostingRegressor


# In[71]:


gbr = GradientBoostingRegressor(max_features=2,max_depth=3, learning_rate=0.01,n_estimators=200)
gbr.fit(X_USA_train,Y_USA_train)


# In[72]:


print(gbr.score(X_USA_train,Y_USA_train))
print(gbr.score(X_USA_test,Y_USA_test))


# In[73]:


feature_importanceGBR=gbr.feature_importances_
sorted_index=np.argsort(feature_importanceGBR)[::-1]
x2=range(len(feature_importanceGBR))
feature_names=df1_USA.columns.tolist()[1:]
labels=np.array(feature_names)[sorted_index]

plt.bar(x2,feature_importanceGBR[sorted_index],tick_label=labels)
plt.xticks(rotation=90);


# # Comparison

# In[69]:


print('------------MEAN SQUARE ERROR (MSE)------------')
print()
print('USA')
print('---------------------')
print('SES: ',mean_squared_error(df_SES_USA['USA'],df_test['USA']))
print('HOLT: ',mean_squared_error(df_Holt_USA['USA'],df_test['USA']))
print('ARIMA: ',mean_squared_error(df_test['USA'],fcast_ARIMA_USA['USA']))
print('SARIMAX',mean_squared_error(fcastSARIMAX_USA,df_test['USA']))
print()

print('POLAND')
print('---------------------')
print('SES: ',mean_squared_error(df_SES_PL['PL'],df_test['PL']))
print('HOLT: ',mean_squared_error(df_Holt_PL['PL'],df_test['PL']))
print('ARIMA: ',mean_squared_error(df_test['PL'],fcast_ARIMA_PL['PL']))
print('SARIMAX',mean_squared_error(fcastSARIMAX_PL,df_test['PL']))
print()

print('TURKEY')
print('---------------------')
print('SES: ',mean_squared_error(df_SES_TR['TR'],df_test['TR']))
print('HOLT: ',mean_squared_error(df_Holt_TR['TR'],df_test['TR']))
print('ARIMA: ',mean_squared_error(df_test['TR'],fcast_ARIMA_TR['TR']))
print('SARIMAX',mean_squared_error(fcastSARIMAX_TR,df_test['TR']))
print()

print('------------MEAN ABSOLUTE ERROR------------')
print()
print('USA')
print('---------------------')
print('SES: ',mean_absolute_error(df_SES_USA['USA'],df_test['USA']))
print('HOLT: ',mean_absolute_error(df_Holt_USA['USA'],df_test['USA']))
print('ARIMA: ',mean_absolute_error(df_test['USA'],fcast_ARIMA_USA['USA']))
print('SARIMAX',mean_absolute_error(fcastSARIMAX_USA,df_test['USA']))
print()

print('POLAND')
print('---------------------')
print('SES: ',mean_absolute_error(df_SES_PL['PL'],df_test['PL']))
print('HOLT: ',mean_absolute_error(df_Holt_PL['PL'],df_test['PL']))
print('ARIMA: ',mean_absolute_error(df_test['PL'],fcast_ARIMA_PL['PL']))
print('SARIMAX',mean_absolute_error(fcastSARIMAX_PL,df_test['PL']))
print()

print('TURKEY')
print('---------------------')
print('SES: ',mean_absolute_error(df_SES_TR['TR'],df_test['TR']))
print('HOLT: ',mean_absolute_error(df_Holt_TR['TR'],df_test['TR']))
print('ARIMA: ',mean_absolute_error(df_test['TR'],fcast_ARIMA_TR['TR']))
print('SARIMAX',mean_absolute_error(fcastSARIMAX_TR,df_test['TR']))
print()


# In[ ]:




