# NAFTRAC-ETF
Investment signal strategy on the NAFTRAC, is an ETF (Exchanged Traded Fund), which replicates the IPC (Index of the Mexican Stock Exchange).

## What is an EFT?
Investing in an ETF is an investment way that must be considered within an investment strategy. ETFs are investment funds that are stock exchange, it is a diversified way of investing that its benchmarking is a particular index, in the case of Naftrac it replicates the IPC, the most important ETF in the US, which is the SPY follows to the S&P500. It is a low cost investment with fewer tax obligations.

# Signal strategy to invest in NAFTRAC
The buy and hold strategy is a passive strategy, since it consists of investing in an asset and maintaining the investment without any movement for a period of time, so that in the end a good return is expected for having waited a long time. It is a strategy that leaves very few benefits. The signal strategy takes the various investment instruments and their available information, continually updated to determine if the financial markets are going up or down; If the markets are on the upside, keep our investment on the asset of interest, and sell the asset at the time that the markdown goes down to obtain good returns.

We will use Indices of the main exchanges in America, Europe and Asia, which are strongly related. It will also allow us to have information in different time zones, which will allow us to have updated information.
The *NAFTRAC* must replicate the index of the Mexican Stock Exchange, which is the *IPC*, we must also consider this as a predictor. The main indexes of the United States such as *S&P500*, *Nasdaq* provide us with information on the trend in the stock markets. In Europe, we take the *FTSE100* Index (Index of the London Stock Exchange) which is in operation at 3:00 a.m. - 11:30 a.m. The Asian indices that we take into account are the *Hang Seng* (HSE) of the Hong Kong stock exchange, *BSE Sensex* (BSE) of the Bombay, India stock exchange, and the *AORD* index of the Australian stock exchange, which are in operation at 8:00 pm - 3:00 am. Similarly, we consider oil as an explanatory variable, predictor, we take the reference prices of the American market, we consider *Oil* as an important variable because Mexico is an oil country, and a large part of its income is given by this commodity. Similarly, we consider the appreciation or depreciation of the *Mexican peso* against the US dollar.
So our equation is as follows:

 **Y =&beta;<sub>0</sub> + &beta;<sub>1</sub>X<sub>1</sub>+&beta;<sub>2</sub>X<sub>2</sub>+&beta;<sub>3</sub>X<sub>3</sub>+...+ &epsilon;**
 
The index information is obtained from Yahoo Finance, through the following library's: `from pandas_datareader import data as pdr`, `pd.core.common.is_list_like = pd.api.types.is_list_like`. Con estas bibliotecas podemos obtener la informacion de todos nuestros predictores, excepto FTSE Index.
To import the London FTSE Index, it is only available `yfinance` package, unfortunately, it is a library that gives us the information with a single decimal, so we will have a bias with the information.
We use other libraries like `pandas`, `numpy`,` statsmodels.formula.api` for our math operations, matrices, regressions, estimating statistics.

# Multiple Linear Regression Model
Our equation can be established as follows

**_Y_=b<sub>0</sub>+b<sub>1</sub>x<sub>1</sub>+b<sub>2</sub>x<sub>2</sub>+b<sub>3</sub>x<sub>3</sub>+...+b<sub>10</sub>x<sub>10</sub>+e**

Where _Y_ is our response variable is tomorrow's NAFTRAC price minus today's opening price. Our predictors, explanatory variables **x <sub>1</sub>,x<sub>2</sub>,x<sub>3</sub>+...+x<sub>10</sub>** are 8 indexes, a commodity and a currency. With our model we seek to make a prediction in the morning, as soon as the Mexican Stock Exchange opens, to determine if we are in a short or long position. Of our 8 indices, a commodity and a currency that are our predictors, we cannot use the information available after they have opened the BMV.

## Indixes Panel
For what we have to use for the Mexican market, as USA market, we will use today's opening price minus yesterday's. Our predictors can be classified into 3 groups. We use the information that Yahoo Finance gives us about the Mexican peso. In the first group will be the markets of Mexico and the US, which will give us information one day late.


|     **Predictors**   |  **Information Available**   |
|----------------------|:----------------------------:|
| *Mexican Markets*    | NAFTRAC: Open-Open Last Day  |
| *Mexican Markets*    | IPC: Open-Open Last Day      |
| *Forex Markets*     | Peso: Open-Open Last Day     |
| *USA Markets*       | Nasdaq: Open-Open Last Day   |
| *USA Markets*       | S&P500: Open-Open Last Day   |
| *USA Mrkets*        | OIL: Open-Open Last Day      |
|                     |                              |
|   *UK Market*       | FTSE: Open-Open Last Day     |
|                     |                              |
| *Asian Markets*     | HSI: Open-Open Last Day      |
| *Asian Markets*     | BSE: Open-Open Last Day      |
| *Asian Markets*     | AORD: Open-Open Last Day     |

The second group also has a day of delay in the information that is the European mark, where we consider the FTSE of the London Stock Exchange. It would also be today's opening price minus yesterday's opening price, that is, the price with which it opened today at dawn minus yesterday's opening price.

The third group is the Asian markets, in which we obtain their daily returns, which is the closing price minus the opening price of the same day, given that the information is already available a few hours before the BMV opens.

The data is obtained from June 2010 to March 25, 2020
We make an Index Panel that gives us the information of our predictors

```
>>> Panel_Indices.tail()
             Naftrac     Naf_A          IPC      Nasdaq       SP500   FTSE         HSI        AORD          BSE       OIL      Peso      Price     
Date
2020-03-19  0.240002 -0.510002  -530.339844   94.130371  -43.020020 -214.3 -512.250000   44.899902   514.871094 -4.480000  0.679399  35.349998     
2020-03-20 -0.830002  0.240002  -281.722656  251.619629   38.459961   71.0  657.730469   44.899902  1455.140625  2.770000  0.389101  35.590000     
2020-03-23 -0.309998 -0.830002  -803.398438 -400.790039 -141.229980   39.2    0.000000  189.199707 -1627.560547  0.049999  0.706799  34.759998     
2020-03-24  0.139999 -0.309998 -1142.921875  348.870117   53.729980 -196.9  165.919922  252.900391  -382.201172  1.430000  0.563601  34.450001     
2020-03-25  0.139999  0.139999  1209.199219  225.209961  113.330078  452.1  271.279297  129.000000  2035.968750  0.690001 -0.542500  34.590000 
```
The variable `Naf_A` is the Naftrac data used one day lag.

### Split Data
We split the data into two parts, we named them `Train` and `Test` respectively. With `Train` is for building the model and `Test` part is for testing model to see if the model can still make reasonable prediction in this dataset.
```python
Train = Panel_Indices.iloc[-2400:-1200, :]
Test = Panel_Indices.iloc[-1200: , :]
print(Train.shape, Test.shape)
```

## Correlation
If we check the scatterplots, which will response NAFTRAC with other ten predictors, there is no expliciit pattern, which is evidence of high noisy properties of stock markets.

 ![Scatter Matrix](https://github.com/SrMich/NAFTRAC-ETF/blob/master/Images/scattermatrix.png ) 


We calculate the correlation between NAFTRAC and the predictors. We can see the relationship more clearly.

```
Naftrac    1.000000
Naf_A     -0.006060
IPC        0.023693
Nasdaq     0.049702
SP500     -0.010664
FTSE      -0.012185
HSI        0.087207
AORD       0.148484
BSE        0.059439
OIL       -0.021273
Peso      -0.138354
Name: Naftrac, dtype: float64
```
We can see that the Asian markets have a greater relationship, because the information they provide us is a few hours lag addition to the fact that the information from the markets of the United States and Mexico is already considered by them.

## OLS Method
We use OLS method of `statsmodels` to build multiple linear equation model.
```python
formula = 'Naftrac~Naf_A+IPC+Nasdaq+SP500+DJI+FTSE+HSI+Nikkei+AORD'
lm = smf.ols(formula=formula, data=Train).fit()
lm.summary()
```

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                Naftrac   R-squared:                       0.055
Model:                            OLS   Adj. R-squared:                  0.047
Method:                 Least Squares   F-statistic:                     6.872
Date:                Fri, 27 Mar 2020   Prob (F-statistic):           1.65e-10
Time:                        18:37:33   Log-Likelihood:                -521.88
No. Observations:                1200   AIC:                             1066.
Df Residuals:                    1189   BIC:                             1122.
Df Model:                          10
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.0135      0.011      1.229      0.219      -0.008       0.035
Naf_A         -0.1190      0.040     -3.011      0.003      -0.197      -0.041
IPC            0.0001    4.7e-05      2.869      0.004    4.26e-05       0.000
Nasdaq         0.0008      0.001      1.527      0.127      -0.000       0.002
SP500         -0.0040      0.002     -2.522      0.012      -0.007      -0.001
FTSE          -0.0002      0.000     -0.986      0.324      -0.001       0.000
HSI            0.0001    6.9e-05      1.659      0.097   -2.09e-05       0.000
AORD           0.0015      0.000      4.677      0.000       0.001       0.002
BSE         2.393e-05   6.09e-05      0.393      0.694   -9.55e-05       0.000
OIL           -0.0072      0.008     -0.898      0.370      -0.023       0.009
Peso          -0.5317      0.122     -4.347      0.000      -0.772      -0.292
==============================================================================
Omnibus:                      125.950   Durbin-Watson:                   2.004
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              787.985
Skew:                           0.238   Prob(JB):                    7.78e-172
Kurtosis:                       6.941   Cond. No.                     4.12e+03
==============================================================================
Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.12e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
```

We analyze the information obtained, the first thing is *Prob(F-statistic):1.65e-10* F test is for overall significance of the multiple linear equation model.

**Significance of the Model - F Test**

**H<sub>0</sub> : &beta;<sub>1</sub> = &beta;<sub>2</sub>= .... = 0** 

**H<sub>a</sub> : at least one of them is not zero**

If we reject, we accept alternative and it means that at least one of the predictors is useful. In the model is better fitted than intercept only model, **P-value= 2.30e-07**, wich is < than 0.05 and it indicates that our model includes useful predictors.
   *P vale = 1.65e-10 < 0.05, Reject H<sub>0</sub>*

Summary table list also list the _P value_ for the train of the significance of the individual predictors

**Significance of the individual predictors - t test**

**H<sub>0</sub>: &beta;<sub>1</sub> = 0             H<sub>0</sub>: &beta;<sub>2</sub> = 0      ....**

**H<sub>a</sub> : &beta;<sub>1</sub>≠0              H<sub>a</sub>: &beta;<sub>2</sub> ≠ 0      ....**

We know that _p value_ if our predictors are significant, we see that  the NAF_A, IPC, AORD and SP500, that means all others predictors are useless information of Naftrac. It may be because of multicollinearity.Also the Mexican currency has a high relationship with the IPC and the Naftrac, given that these instruments are very sensitive to the appreciation of the dollar, given that the US is its main trading partner. This generates a high multicollinearity. Multicollinearity refers to a situation in which two or more predictors in the multiple regression model are highly, linearly related. All indices of different markets are correlated. Multicollinearity does not reduce predictive power. In fact our model has a problem of strong multicollinearity between the IPC and Naf_A predictors, this is because the EFT follows, it replicates the IPC.

## Making Prediction
Now we can predict daily change of *Naftrac* using mmesagge predict of our model _lm_.

```python
Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)
```
We predict NAFTRAC in both _train_ and _test_, We can see the scatterplot between the Train and Test daily change, and predict the NAFTRAC daily change. 
It does have positive correlation although no very strong. Considering it is daily change, this result is not very band

 ![Prediction](https://github.com/SrMich/NAFTRAC-ETF/blob/master/Images/Test_Train.png "TrainTest ") 

### Model Evaluation
First statistic is RMSE, which is the square root of sum of squared errors averaged by degrees of freedom where _k_ is number of predictors. 
This statistic is to measure the prediction error.

<img src = "https://github.com/SrMich/NAFTRAC-ETF/blob/master/Images/RMSE.png" width="250">

The second is adjusted R-square. In simple Linear Regression, we use R-square to get the percentage of variation that can be explained by a model. 
We found that by adding more predictors the askew is always increasing but the accurracy is even worse. To compensate the effects of numbers predictors, we have adjusted R-square, which measures percentage of variation of a response that is explained by the model. 
We compared to R^2 and RMSE between train and test and we check whether they are different dramatically. If so, this called overfitting.
We compute R-squared and RMSE in our model.

|         | **Train**  | **Test**  |
|---------|:----------:|:----------|
| **R^2** |  0.046691  | 0.071941  |
|**RMSE** |  0.375523  | 0.43357   |

RMSE and adjusted R^2 is much better in train than in test dataset. RMSE increase in test which is a bit worse than tha in the train. Our model is not overfitted. Our R^2 is only 7.19% percent, which is quite low, but in stock market it's not that bad.

# Evaluate the strategy
Now we will use predict price change of Naftrac as trading signal and then perform a simple strategy. If the signal is positive, we are *Long*. Otherwise, we are *Short*.

## Profit of Signal-Based Strategy (Train data)
First, we will compute a position of our trading based on our predicted value of a response. `Order=1` if predicted value is positive or our prediction for price change is positive for opening today to opening tomorrow. 
Otherwise, `Order=-1` which means we will sell one share if we had one share, and then short sell one share. Daily profit is computed in column is profit. Finally, we compute a cumulative profit and sum it in the column of wealth

```python
Train['Order'] = [1 if sig>0 else -1 for sig in Train['PredictedY']]
Train['Profit'] = Train['Naftrac'] * Train['Order']
Train['Wealth'] = Train['Profit'].cumsum()
print('Total profit (Train data): ', Train['Profit'].sum())

Total profit (Train data):  77.03001022338867
```
**Total profit in this 1200 days is $77**

Then , we can compare the performance of this strategy, which we call the signal-based strategy with a passive strategy, which we call buy and hold strategy, which is to buy more shares of Nafctrac inially and hold it for 12000 days.

![Strategy](https://github.com/SrMich/NAFTRAC-ETF/blob/master/Images/Strategy_Train.png "Strategy Train")

## Profit of Signal-Based Strategy (Test data)
Similarly, we can view it as trading in test dataset.

```python
Test['Order'] = [1 if sig>0 else -1 for sig in Test['PredictedY']]
Test['Profit'] = Test['Naftrac'] * Test['Order']
Test['Wealth'] = Test['Profit'].cumsum()
print('Total Profit (Test): ', Test['Profit'].sum())

         Total Profit (Test):  104.29998397827148
```
The total profit is $104, is greather than in train

![Strategy](https://github.com/SrMich/NAFTRAC-ETF/blob/master/Images/Strategy_Test.png "Strategy Train")

The consistency of performance is very important. Otherwise, it is too risky to apply it in the future

# Evaluating of model practical standard
Average daily return is amirror we can make comparison in Finance Industry when they use a _Sharpe Ratio_ and _Maximum Drawdown_.

+ Sharpe Ratio
+ Maximum Drawdown

## Sharpe Ratio
*Sharpe Ratio* measures excess return (or risk premium) per unit of deviation in an investment asset or a trading strategy, tipically referred to as risk. For example, daily Sharpe ratio is equal to the mean of excess return divided by standard deviatio of excess return. Since there are about 252 trading days per year in Mexico stock market.

<img src = "https://github.com/SrMich/NAFTRAC-ETF/blob/master/Images/SharpeRatio.png" width="500">

We compute the Sharpe Ratio, we first need to revise wealth process by including initial investment which is the price of one share of Naftrac.

```python
Train['Wealth'] = Train['Wealth'] + Train.loc[Train.index[0], 'Price']
Test['Wealth'] = Test['Wealth'] + Test.loc[Test.index[0], 'Price']
Test['Return'] = np.log(Test['Wealth']) - np.log(Test['Wealth'].shift(1))
RendDailyTest = Test['Return'].dropna()
SharpeRatioTest = RendDailyTest.mean()/RendDailyTest.std(ddof=1)
SharpeRatioTest_Anual = np.sqrt(252)*RendDailyTest.mean()/RendDailyTest.std(ddof=1)
```
|                          | **Train**  |  **Test**|
|--------------------------|:----------:|:---------|
|**Daily Sharpe Ratio**    | 0.1552     | 0.2024   |
|**Yearly Sharpe Ratio**   | 2.4634     | 3.2129   |
         
Yearly Sharpe Ratio is 2.46 for trainig data and 3.21 for test data which is greather than of the train data.

## Maximum Drawdown
The Maximum Drawdown is a maximum percentage decline in the strategy from the historical peak profit at each point in time.

<img src = "https://github.com/SrMich/NAFTRAC-ETF/blob/master/Images/Drawdown.png" width="280">

We compute drawdown, and then the maximum of all drawdowns in the trading period. Maxim drawdown is that risk of mirror for extreme loss a strategy.

|                      | **Train**  | **Test** |
|----------------------|:----------:|:---------|
| **Maximum Drawdown** |   0.0665   |   0.085  |

We find Maximum Drawdown in train and in test are close. The result shows that if we apply this strategy in test set, the maximum loss from the peak is 8.5%.

From the mirror of Sharpe Ratio and Maximum Drawdown we can tell that the performance of strategy is quite consistent in place of extreme loss, and the return per unit risk is also consistent.

We conclude that the derived model is not overfitted and the performance of signal-based strategy is consistent in termns of large loss and returns.
