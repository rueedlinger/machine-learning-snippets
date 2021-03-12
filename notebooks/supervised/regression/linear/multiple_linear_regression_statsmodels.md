>**Note**: This is a generated markdown export from the Jupyter notebook file [multiple_linear_regression_statsmodels.ipynb](multiple_linear_regression_statsmodels.ipynb).

## Linear regression with statsmodels (OLS)


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools.eval_measures as eval_measures
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import datasets, model_selection, metrics
```

We load the boston house-prices dataset and `X` are our features and `y` is the target variable `medv` (Median value of owner-occupied homes in $1000s).


```python
boston = datasets.load_boston()
print(boston.DESCR)
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    


Let's split the data in a test and training set.


```python
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7)
```


```python
df_train = pd.DataFrame(y_train, columns=['target'])
df_train['type'] = 'train'

df_test = pd.DataFrame(y_test, columns=['target'])
df_test['type'] = 'test'

df_set = df_train.append(df_test)

_ = sns.displot(df_set, x="target" ,hue="type", kind="kde", log_scale=False)
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_6_0.png)
    


## Fitting models - the standard way
### Full model without an intercept



```python
model = sm.OLS(y_train, X_train)
result = model.fit()
print(result.summary())
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                      y   R-squared (uncentered):                   0.956
    Model:                            OLS   Adj. R-squared (uncentered):              0.954
    Method:                 Least Squares   F-statistic:                              567.7
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):                   5.43e-222
    Time:                        14:44:49   Log-Likelihood:                         -1080.2
    No. Observations:                 354   AIC:                                      2186.
    Df Residuals:                     341   BIC:                                      2237.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1123      0.038     -2.972      0.003      -0.187      -0.038
    ZN             0.0596      0.019      3.216      0.001       0.023       0.096
    INDUS         -0.0022      0.077     -0.029      0.977      -0.155       0.150
    CHAS           2.1291      1.124      1.894      0.059      -0.082       4.341
    NOX            1.3086      4.288      0.305      0.760      -7.125       9.742
    RM             5.4903      0.384     14.315      0.000       4.736       6.245
    AGE            0.0059      0.018      0.337      0.736      -0.029       0.040
    DIS           -0.9076      0.253     -3.589      0.000      -1.405      -0.410
    RAD            0.1723      0.084      2.052      0.041       0.007       0.338
    TAX           -0.0093      0.005     -1.899      0.058      -0.019       0.000
    PTRATIO       -0.3646      0.142     -2.568      0.011      -0.644      -0.085
    B              0.0151      0.003      4.740      0.000       0.009       0.021
    LSTAT         -0.5190      0.065     -7.964      0.000      -0.647      -0.391
    ==============================================================================
    Omnibus:                      138.221   Durbin-Watson:                   1.935
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              856.792
    Skew:                           1.505   Prob(JB):                    8.91e-187
    Kurtosis:                      10.002   Cond. No.                     8.73e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.73e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_9_0.png)
    



```python
result.pvalues < 0.05
```




    CRIM        True
    ZN          True
    INDUS      False
    CHAS       False
    NOX        False
    RM          True
    AGE        False
    DIS         True
    RAD         True
    TAX        False
    PTRATIO     True
    B           True
    LSTAT       True
    dtype: bool




```python
predicted = result.predict(X_test)

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.7058520892278357
    mse: 20.567935033717376
    rmse: 4.535188533425857
    mae: 3.300625707951339


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.748
    Model:                            OLS   Adj. R-squared:                  0.738
    Method:                 Least Squares   F-statistic:                     77.47
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           3.27e-93
    Time:                        14:44:49   Log-Likelihood:                -1056.2
    No. Observations:                 354   AIC:                             2140.
    Df Residuals:                     340   BIC:                             2195.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         43.8245      6.239      7.024      0.000      31.553      56.096
    CRIM          -0.1248      0.035     -3.523      0.000      -0.194      -0.055
    ZN             0.0551      0.017      3.178      0.002       0.021       0.089
    INDUS          0.0360      0.073      0.495      0.621      -0.107       0.179
    CHAS           1.8485      1.053      1.756      0.080      -0.223       3.920
    NOX          -18.5071      4.905     -3.773      0.000     -28.155      -8.859
    RM             3.2741      0.478      6.851      0.000       2.334       4.214
    AGE            0.0154      0.016      0.933      0.352      -0.017       0.048
    DIS           -1.5242      0.252     -6.038      0.000      -2.021      -1.028
    RAD            0.3482      0.083      4.220      0.000       0.186       0.510
    TAX           -0.0131      0.005     -2.854      0.005      -0.022      -0.004
    PTRATIO       -1.1353      0.172     -6.588      0.000      -1.474      -0.796
    B              0.0089      0.003      2.889      0.004       0.003       0.015
    LSTAT         -0.6144      0.062     -9.833      0.000      -0.737      -0.492
    ==============================================================================
    Omnibus:                      123.521   Durbin-Watson:                   1.986
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              519.615
    Skew:                           1.460   Prob(JB):                    1.47e-113
    Kurtosis:                       8.168   Cond. No.                     1.55e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.55e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")

```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_14_0.png)
    



```python
predicted = result.predict(sm.add_constant(X_test))

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.7030418964706355
    mse: 20.764434345613164
    rmse: 4.556800889397425
    mae: 3.358118810448314


## Fitting models using R-style formulas
We can also fit a model with the R syntax `y ~ x_1 + x_2` and build some complexer models.


```python
dat = X_train.copy()
dat['MEDV'] = y_train
dat.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>456</th>
      <td>4.66883</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>5.976</td>
      <td>87.9</td>
      <td>2.5806</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>10.48</td>
      <td>19.01</td>
      <td>12.7</td>
    </tr>
    <tr>
      <th>245</th>
      <td>0.19133</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.431</td>
      <td>5.605</td>
      <td>70.2</td>
      <td>7.9549</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>389.13</td>
      <td>18.46</td>
      <td>18.5</td>
    </tr>
    <tr>
      <th>167</th>
      <td>1.80028</td>
      <td>0.0</td>
      <td>19.58</td>
      <td>0.0</td>
      <td>0.605</td>
      <td>5.877</td>
      <td>79.2</td>
      <td>2.4259</td>
      <td>5.0</td>
      <td>403.0</td>
      <td>14.7</td>
      <td>227.61</td>
      <td>12.14</td>
      <td>23.8</td>
    </tr>
    <tr>
      <th>343</th>
      <td>0.02543</td>
      <td>55.0</td>
      <td>3.78</td>
      <td>0.0</td>
      <td>0.484</td>
      <td>6.696</td>
      <td>56.4</td>
      <td>5.7321</td>
      <td>5.0</td>
      <td>370.0</td>
      <td>17.6</td>
      <td>396.90</td>
      <td>7.18</td>
      <td>23.9</td>
    </tr>
    <tr>
      <th>224</th>
      <td>0.31533</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.504</td>
      <td>8.266</td>
      <td>78.3</td>
      <td>2.8944</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>385.05</td>
      <td>4.14</td>
      <td>44.8</td>
    </tr>
  </tbody>
</table>
</div>



### Full model with an intercept


```python
result = smf.ols('MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                   MEDV   R-squared:                       0.676
    Model:                            OLS   Adj. R-squared:                  0.664
    Method:                 Least Squares   F-statistic:                     59.24
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           7.21e-76
    Time:                        14:44:49   Log-Likelihood:                -1100.5
    No. Observations:                 354   AIC:                             2227.
    Df Residuals:                     341   BIC:                             2277.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     30.4908      6.892      4.424      0.000      16.935      44.046
    CRIM          -0.1757      0.040     -4.433      0.000      -0.254      -0.098
    ZN             0.0455      0.020      2.323      0.021       0.007       0.084
    INDUS         -0.0271      0.082     -0.330      0.741      -0.188       0.134
    CHAS           1.7674      1.192      1.483      0.139      -0.576       4.111
    NOX          -25.3702      5.494     -4.618      0.000     -36.177     -14.563
    RM             5.6719      0.465     12.195      0.000       4.757       6.587
    AGE           -0.0351      0.018     -1.981      0.048      -0.070      -0.000
    DIS           -1.7076      0.285     -5.995      0.000      -2.268      -1.147
    RAD            0.3416      0.093      3.659      0.000       0.158       0.525
    TAX           -0.0131      0.005     -2.512      0.012      -0.023      -0.003
    PTRATIO       -1.2488      0.195     -6.418      0.000      -1.631      -0.866
    B              0.0134      0.003      3.873      0.000       0.007       0.020
    ==============================================================================
    Omnibus:                      194.923   Durbin-Watson:                   1.910
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1713.133
    Skew:                           2.154   Prob(JB):                         0.00
    Kurtosis:                      12.878   Cond. No.                     1.54e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.54e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_20_0.png)
    



```python
predicted = result.predict(X_test)

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.6912849930900166
    mse: 21.586521520380032
    rmse: 4.646129735638043
    mae: 3.333816918058118


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.685
    Model:                            OLS   Adj. R-squared:                  0.676
    Method:                 Least Squares   F-statistic:                     74.52
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           8.65e-80
    Time:                        14:44:50   Log-Likelihood:                 5.0609
    No. Observations:                 354   AIC:                             11.88
    Df Residuals:                     343   BIC:                             54.44
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.6553      0.303     12.061      0.000       3.059       4.251
    CRIM           -0.0126      0.002     -7.272      0.000      -0.016      -0.009
    CHAS            0.0698      0.052      1.343      0.180      -0.032       0.172
    NOX            -1.2192      0.235     -5.187      0.000      -1.682      -0.757
    RM              0.1999      0.020     10.055      0.000       0.161       0.239
    DIS            -0.0534      0.011     -4.852      0.000      -0.075      -0.032
    RAD             0.0155      0.004      3.901      0.000       0.008       0.023
    TAX            -0.0006      0.000     -3.010      0.003      -0.001      -0.000
    PTRATIO        -0.0538      0.008     -6.766      0.000      -0.069      -0.038
    B               0.0006      0.000      4.165      0.000       0.000       0.001
    pow(AGE, 2)  -1.63e-05   6.42e-06     -2.539      0.012   -2.89e-05   -3.67e-06
    ==============================================================================
    Omnibus:                       81.051   Durbin-Watson:                   1.986
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              526.026
    Skew:                           0.767   Prob(JB):                    5.96e-115
    Kurtosis:                       8.772   Cond. No.                     1.77e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.77e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.


Let's plot the QQ-Plot for the residuals


```python
result.pvalues < 0.05
```




    Intercept       True
    CRIM            True
    CHAS           False
    NOX             True
    RM              True
    DIS             True
    RAD             True
    TAX             True
    PTRATIO         True
    B               True
    pow(AGE, 2)     True
    dtype: bool




```python
predicted = np.exp(result.predict(X_test))

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.6862238117383626
    mse: 21.9404184729758
    rmse: 4.684060041563921
    mae: 3.073112663458295



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
