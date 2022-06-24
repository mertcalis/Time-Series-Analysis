# Time Series Analysis Methods
## Models

- Simple Exponential Smoothing (SES)
- Holt
- Holt-Winter's Method
- Autoregressive integrated moving average (ARIMA)
- SARIMA/SARIMAX
- Random Forest Tree Regression
- Gradient Boosted Trees

## Features

- Import a CSV file and convert it to Data Frame
- Setting index and frequency
- Seasonality,Stationary Check
    >seasonal_decompose(model="Additive")
    >Augmented Dickey-Fuller test (Adfuller test)
- Implementations of clasical time series analysis methods and forecasting
- Implementations of machine learning methods, grid search and forecasting
- Plotting the results
- Comparison of results (Mean square error, Mean absolute error)



## Frameworks

- [Numpy] 
- [statsmodels] - provides functions for the estimation of many different statistical models, statistical tests, statistical data exploration.
- [sklearn] - various classification, regression and clustering algorithms including support-vector machines, random forests, gradient boosting
- [Pandas] -  is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the Python programming language.

## Data
Collected from Google Trends
https://trends.google.com/trends/explore?date=all&geo=US&q=nba

import data via pandas

```sh
rawUsa = pd.read_csv('NbaUsa.csv',skiprows=1,parse_dates=True)
df = rawPl.merge(rawUsa, on="Ay").merge(rawTr, on="Ay")
df["Date"] = df["Ay"].astype(str)
df["Date"]=pd.to_datetime(df["Date"])
df.set_index('Date',inplace=True)
df.index.freq ='MS'
df.drop('Ay',inplace=True,axis=1)
df.head()
```
![image](https://user-images.githubusercontent.com/44343742/175719364-ff634292-9134-473a-bdd7-6272be81af88.png)
```sh
df['USA'].plot()
```
![image](https://user-images.githubusercontent.com/44343742/175719145-0cceab50-5e88-4e99-97ca-5daf16b579ca.png)
## Seasonal Decompose and Adfuller Test
```sh
decompose_dataUSA = seasonal_decompose(df['USA'],model="Additive")
decompose_dataUSA.plot();
```
![image](https://user-images.githubusercontent.com/44343742/175720068-1d97b785-6dc5-4e91-82e8-26eb470fd6ea.png)

```sh
dfadfTestUSA = adfuller(diffUSA.dropna(), autolag = 'AIC')
print("--USA--")
print("1. ADF : ",dfadfTestUSA[0])
print("2. P-Value : ", dfadfTestUSA[1])
print("3. Num Of Lags : ", dfadfTestUSA[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dfadfTestUSA[3])
print("5. Critical Values :")
for key, val in dfadfTestUSA[4].items():
    print("\t",key, ": ", val)
```

![image](https://user-images.githubusercontent.com/44343742/175720200-6b54d6d1-033e-480c-84fd-f2b0ad33cee2.png)

## Prepare Data for Training
```sh
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
```
![image](https://user-images.githubusercontent.com/44343742/175720342-9da476be-f28e-441c-8e02-ba830d1df92f.png)
## Results

### Simple Exponential Smoothing (SES) and HOLT Method

Simple Exponential Smoothing, is a time series forecasting method for univariate data without a trend or seasonality.
It requires a single parameter, called alpha (a), also called the smoothing factor or smoothing coefficient.

Holt is an exponential smoothing method by performing weighting on past observations and estimate trend in data.
It requires two parameters. In the weighting parameters, the new value has a greater than previous observations.

![image](https://user-images.githubusercontent.com/44343742/175720668-27337114-5004-4313-8888-1fa41d5b8196.png)

### Holt-Winter's Method
![image](https://user-images.githubusercontent.com/44343742/175720803-a4457fa3-47e2-4e56-9370-16182d865050.png)

### Autoregressive integrated moving average (ARIMA)
![image](https://user-images.githubusercontent.com/44343742/175720863-d448d161-66a6-4dc6-9fff-47b41220e951.png)

### SARIMA/SARIMAX
![image](https://user-images.githubusercontent.com/44343742/175720900-97da9fb1-d067-4860-bafe-fe306d84187f.png)

### Random Forest
Grid search
```sh
from sklearn.model_selection import ParameterGrid
grid ={'n_estimators': range(20,200,10),'max_depth': [2,3,4,5,6,7,8,9,10],'max_features':[2,3,4,5,6,7,8,9], 'random_state' : [18,19]}

rfr = RandomForestRegressor()
test_scores = []

for g in ParameterGrid(grid):
    rfr.set_params(**g)
    rfr.fit(X_PL_train,Y_PL_train)
    test_scores.append(rfr.score(X_PL_test,Y_PL_test))
    
best_idx = np.argmax(test_scores)
print(test_scores[best_idx],ParameterGrid(grid)[best_idx])
```
![image](https://user-images.githubusercontent.com/44343742/175721075-c0f13dc3-b9cf-4d21-9529-2985b171c79d.png)

![image](https://user-images.githubusercontent.com/44343742/175721081-68c73d4e-4219-4b09-979a-c4ef140ecd3a.png)

### Gradient Boosted Trees
Applied same coefficients
![image](https://user-images.githubusercontent.com/44343742/175721228-a591085b-f669-413d-a603-f7f1c0ef26e2.png)

## Comparison
![image](https://user-images.githubusercontent.com/44343742/175721410-eacc545d-d0e9-47f3-961d-7dd130ef6764.png)
![image](https://user-images.githubusercontent.com/44343742/175721454-1fc6c665-e6d0-4d8e-97f8-4c6d6cdc6b6b.png)

