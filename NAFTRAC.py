import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
pd.core.common.is_list_like = pd.api.types.is_list_like
import datetime

import warnings
warnings.filterwarnings("ignore")

# Descargar datos
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2020,3,25)

### México
NAFTRAC = pdr.get_data_yahoo("NAFTRACISHRS.MX", start , end) # ETF que replica el indice de la Bolsa Mexicana de Valores (BMV)
NAFTRAC.head()
IPC = pdr.get_data_yahoo("^MXX", start, end) # México BMV
IPC.head()

# Estados Unidos
Nasdaq = pdr.get_data_yahoo("^IXIC", start, end)
Nasdaq.head()
SP500 = pdr.get_data_yahoo("^GSPC", start, end)
SP500.head()

# Asia
HSI = pdr.get_data_yahoo("^HSI", start, end) # Hong Kong
HSI.head()
AORD = pdr.get_data_yahoo("^AORD", start, end) # Australia
AORD.head()
BSE = pdr.get_data_yahoo("^BSESN", start, end)
BSE.head()

### Europa
FTSE = yf.download("^FTSE", start = "2010-01-01", end = "2020-03-26") # Londres
FTSE.head()

## Otros
OIL = pdr.get_data_yahoo("CL=F", start, end)
OIL.head()
Peso = pdr.get_data_yahoo("MXN=X", start, end)
Peso.head()

# Indice Panel
Panel_Indices = pd.DataFrame(index=NAFTRAC.index)

Panel_Indices['Naftrac'] = NAFTRAC['Open'].shift(-1) - NAFTRAC['Open']
Panel_Indices['Naf_A'] = NAFTRAC['Open'] - NAFTRAC['Open'].shift(1)
Panel_Indices['IPC'] = IPC['Open'] - IPC['Open'].shift(1)

# EUA
Panel_Indices['Nasdaq'] = Nasdaq['Open'] - Nasdaq['Open'].shift(1)
Panel_Indices['SP500'] = SP500['Open'] - SP500['Open'].shift(1)

# Europa
Panel_Indices['FTSE'] = FTSE['Open'] - FTSE['Open'].shift(1)
# Asia
Panel_Indices['HSI'] = HSI['Close'] - HSI['Open']
Panel_Indices['AORD'] = AORD['Close'] - AORD['Open']
Panel_Indices['BSE'] = BSE['Close'] - BSE['Open']
##
Panel_Indices['OIL'] = OIL['Open'] - OIL['Open'].shift(1)
Panel_Indices['Peso'] = Peso['Open'] - Peso['Open'].shift(1)

Panel_Indices['Price'] = NAFTRAC['Open']

Panel_Indices.head()

Panel_Indices.isnull().sum()
Panel_Indices = Panel_Indices.fillna(method='ffill')
Panel_Indices = Panel_Indices.dropna()
Panel_Indices.isnull().sum()

Panel_Indices.tail()

# Dividir datos
print(Panel_Indices.shape)
Train = Panel_Indices.iloc[-2400:-1200, :]
Test = Panel_Indices.iloc[-1200: , :]
print(Train.shape, Test.shape)

# Exploración del conjunto de datos Train
from pandas.plotting import scatter_matrix
md = scatter_matrix(Train, figsize=(12, 12))
plt.show(md)

# Análisis  de la correlación de cada indice respecto a Naftrac
CorrelacionNaf = Train.iloc[:, :-1].corr()['Naftrac']
print(CorrelacionNaf)

formula = 'Naftrac~Naf_A+IPC+Nasdaq+SP500+FTSE+HSI+AORD+BSE+OIL+Peso'
lm = smf.ols(formula=formula, data=Train).fit()
lm.summary()

# Realización de la predicción
Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)

plt.scatter(Train['Naftrac'], Train['PredictedY'])
plt.title("Tran vs Test")
plt.xlabel("Train")
plt.ylabel("Test")
plt.grid(True)
plt.show()

## Valuación del modelo
# Parametros: Root Mean Squared Error, Adjusted R^2
def adjustedMetric(data, modelo, model_k, yname):
    data['yhat'] = modelo.predict(data)
    SST = ((data[yname] - data[yname].mean())**2).sum()
    SSR = ((data['yhat'] - data[yname].mean())**2).sum()
    SSE = ((data[yname] - data['yhat'])**2).sum()
    r2 = SSR/SST
    adjustR2 = 1 - (1-r2)*(data.shape[0] - 1)/(data.shape[0] -model_k -1)
    RMSE = (SSE/(data.shape[0] -model_k -1))**0.5
    return adjustR2, RMSE

def TableEvaluation(test, train, modelo, model_k, yname):
    r2test, RMSEtest = adjustedMetric(test, modelo, model_k, yname)
    r2train, RMSEtrain = adjustedMetric(train, modelo, model_k, yname)
    evaluacion = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    evaluacion['Train'] = [r2train, RMSEtrain]
    evaluacion['Test'] = [r2test, RMSEtest]
    return evaluacion

# Conocer si el modelo eno est muy ajustado
TableEvaluation(Test, Train, lm, 10, 'Naftrac')

# Estrategia basado en señales
# Train
Train['Order'] = [1 if sig>0 else -1 for sig in Train['PredictedY']]
Train['Profit'] = Train['Naftrac'] * Train['Order']

Train['Wealth'] = Train['Profit'].cumsum()
print('Total Profit (Train data): ', Train['Profit'].sum())

plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy (Train)')
plt.plot(Train['Wealth'].values, color='blue', label='Signal Based Strategy')
plt.plot(Train['Naftrac'].cumsum().values, color='red', label='Buy and Hold Strategy')
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
plt.show()

## Test
Test['Order'] = [1 if sig>0 else -1 for sig in Test['PredictedY']]
Test['Profit'] = Test['Naftrac'] * Test['Order']

Test['Wealth'] = Test['Profit'].cumsum()
print('Total Profit (Test): ', Test['Profit'].sum())

plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy (Test)')
plt.plot(Test['Wealth'].values, color='green', label='Signal Based Strategy')
plt.plot(Test['Naftrac'].cumsum().values, color='red', label='Buy and Hold Strategy')
plt.legend()
plt.ylabel("Cumulative Returns")
plt.grid(True)
plt.show()


## Evaluación del modelo
Train['Wealth'] = Train['Wealth'] + Train.loc[Train.index[0], 'Price']
Test['Wealth'] = Test['Wealth'] + Test.loc[Test.index[0], 'Price']

# Sharpe Ratio en los datos de prueba (Train)
Train['Returns'] = np.log(Train['Wealth']) - np.log(Train['Wealth'].shift(1))
RendDailyTrain = Train['Returns'].dropna()
SharpeRatioTrain = RendDailyTrain.mean()/RendDailyTrain.std(ddof=1)
SharpeRatioTrain_Anual = np.sqrt(252)*RendDailyTrain.mean()/RendDailyTrain.std(ddof=1)

print('Daily Sharpe Ratio (Train) is ', SharpeRatioTrain)
print('Yearly Sherpe Ratio (Train) is ', SharpeRatioTrain_Anual)

# Sharpe Ratio en los datos del Test
Test['Return'] = np.log(Test['Wealth']) - np.log(Test['Wealth'].shift(1))
RendDailyTest = Test['Return'].dropna()

SharpeRatioTest = RendDailyTest.mean()/RendDailyTest.std(ddof=1)
SharpeRatioTest_Anual = np.sqrt(252)*RendDailyTest.mean()/RendDailyTest.std(ddof=1)

print('Daily Sharpe Ratio (Test) is ', SharpeRatioTest) 
print('Yearly Sherpe Ratio (Test) is ', SharpeRatioTest_Anual)

## # Maximum Drawdown en los datos de prueba (Train)
Train['Peak'] = Train['Wealth'].cummax()
Train['Drawdown'] = (Train['Peak'] - Train['Wealth'])/Train['Peak']
print('Maximum Drawdown (Train) is ', Train['Drawdown'].max())

# Maximum Drawdown en los datos Test
Test['Peak'] = Test['Wealth'].cummax()
Test['Drawdown'] = (Test['Peak'] - Test['Wealth'])/Test['Peak']
print('Maximum Drawdown (Test) is ', Test['Drawdown'].max())
