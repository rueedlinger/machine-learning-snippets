>**Note**: This is a generated markdown export from the Jupyter notebook file [multiple_linear_regression_statsmodels.ipynb](multiple_linear_regression_statsmodels.ipynb).

## Linear regression with statsmodels (OLS)


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools.eval_measures as eval_measures
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

## Fitting models - the standard way
### Full model without an intercept



```python
model = sm.OLS(y_train, X_train)
result = model.fit()
print(result.summary())
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                      y   R-squared (uncentered):                   0.964
    Model:                            OLS   Adj. R-squared (uncentered):              0.962
    Method:                 Least Squares   F-statistic:                              696.8
    Date:                Tue, 09 Mar 2021   Prob (F-statistic):                   1.56e-236
    Time:                        17:40:14   Log-Likelihood:                         -1046.5
    No. Observations:                 354   AIC:                                      2119.
    Df Residuals:                     341   BIC:                                      2169.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0771      0.035     -2.185      0.030      -0.147      -0.008
    ZN             0.0283      0.017      1.710      0.088      -0.004       0.061
    INDUS         -0.0067      0.069     -0.097      0.923      -0.143       0.129
    CHAS           1.9403      0.990      1.960      0.051      -0.007       3.888
    NOX           -7.4063      4.034     -1.836      0.067     -15.341       0.529
    RM             6.9329      0.358     19.347      0.000       6.228       7.638
    AGE           -0.0253      0.016     -1.589      0.113      -0.057       0.006
    DIS           -0.9505      0.228     -4.167      0.000      -1.399      -0.502
    RAD            0.1655      0.084      1.964      0.050      -0.000       0.331
    TAX           -0.0100      0.005     -2.057      0.040      -0.020      -0.000
    PTRATIO       -0.5538      0.126     -4.401      0.000      -0.801      -0.306
    B              0.0120      0.003      3.809      0.000       0.006       0.018
    LSTAT         -0.2749      0.060     -4.546      0.000      -0.394      -0.156
    ==============================================================================
    Omnibus:                      141.833   Durbin-Watson:                   2.170
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1220.730
    Skew:                           1.433   Prob(JB):                    8.36e-266
    Kurtosis:                      11.634   Cond. No.                     8.92e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.92e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_8_0.png)
    



```python
result.pvalues < 0.05
```




    CRIM        True
    ZN         False
    INDUS      False
    CHAS       False
    NOX        False
    RM          True
    AGE        False
    DIS         True
    RAD        False
    TAX         True
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

    r2 score: 0.6087344802399262
    mse: 33.046982164853965
    rmse: 5.748650464661594
    mae: 3.5012280679876837


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.760
    Model:                            OLS   Adj. R-squared:                  0.751
    Method:                 Least Squares   F-statistic:                     82.73
    Date:                Tue, 09 Mar 2021   Prob (F-statistic):           7.95e-97
    Time:                        17:40:14   Log-Likelihood:                -1034.9
    No. Observations:                 354   AIC:                             2098.
    Df Residuals:                     340   BIC:                             2152.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         29.1513      6.075      4.799      0.000      17.203      41.100
    CRIM          -0.0931      0.034     -2.709      0.007      -0.161      -0.025
    ZN             0.0277      0.016      1.729      0.085      -0.004       0.059
    INDUS          0.0123      0.067      0.182      0.855      -0.120       0.144
    CHAS           1.5538      0.963      1.614      0.108      -0.340       3.448
    NOX          -18.4673      4.539     -4.069      0.000     -27.395      -9.540
    RM             5.1771      0.504     10.262      0.000       4.185       6.169
    AGE           -0.0187      0.016     -1.204      0.230      -0.049       0.012
    DIS           -1.3731      0.238     -5.771      0.000      -1.841      -0.905
    RAD            0.2905      0.086      3.388      0.001       0.122       0.459
    TAX           -0.0130      0.005     -2.720      0.007      -0.022      -0.004
    PTRATIO       -0.9649      0.149     -6.474      0.000      -1.258      -0.672
    B              0.0069      0.003      2.115      0.035       0.000       0.013
    LSTAT         -0.3882      0.063     -6.144      0.000      -0.512      -0.264
    ==============================================================================
    Omnibus:                      141.377   Durbin-Watson:                   2.118
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              912.484
    Skew:                           1.532   Prob(JB):                    7.19e-199
    Kurtosis:                      10.244   Cond. No.                     1.53e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.53e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_13_0.png)
    



```python
predicted = result.predict(sm.add_constant(X_test))

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.6610844335083361
    mse: 28.625411940488345
    rmse: 5.350272137049512
    mae: 3.5187802491122837


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
      <th>451</th>
      <td>5.44114</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.655</td>
      <td>98.2</td>
      <td>2.3552</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>355.29</td>
      <td>17.73</td>
      <td>15.2</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1.35472</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.072</td>
      <td>100.0</td>
      <td>4.1750</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>376.73</td>
      <td>13.04</td>
      <td>14.5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.11747</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>6.009</td>
      <td>82.9</td>
      <td>6.2267</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>396.90</td>
      <td>13.27</td>
      <td>18.9</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.11027</td>
      <td>25.0</td>
      <td>5.13</td>
      <td>0.0</td>
      <td>0.453</td>
      <td>6.456</td>
      <td>67.8</td>
      <td>7.2255</td>
      <td>8.0</td>
      <td>284.0</td>
      <td>19.7</td>
      <td>396.90</td>
      <td>6.73</td>
      <td>22.2</td>
    </tr>
    <tr>
      <th>196</th>
      <td>0.04011</td>
      <td>80.0</td>
      <td>1.52</td>
      <td>0.0</td>
      <td>0.404</td>
      <td>7.287</td>
      <td>34.1</td>
      <td>7.3090</td>
      <td>2.0</td>
      <td>329.0</td>
      <td>12.6</td>
      <td>396.90</td>
      <td>4.08</td>
      <td>33.3</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.733
    Model:                            OLS   Adj. R-squared:                  0.724
    Method:                 Least Squares   F-statistic:                     78.06
    Date:                Tue, 09 Mar 2021   Prob (F-statistic):           4.30e-90
    Time:                        17:40:14   Log-Likelihood:                -1053.5
    No. Observations:                 354   AIC:                             2133.
    Df Residuals:                     341   BIC:                             2183.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     15.2019      5.930      2.563      0.011       3.537      26.867
    CRIM          -0.1132      0.036     -3.144      0.002      -0.184      -0.042
    ZN             0.0200      0.017      1.190      0.235      -0.013       0.053
    INDUS         -0.0199      0.071     -0.282      0.778      -0.159       0.119
    CHAS           1.7531      1.013      1.731      0.084      -0.239       3.745
    NOX          -19.4927      4.774     -4.083      0.000     -28.882     -10.103
    RM             7.0025      0.429     16.318      0.000       6.158       7.847
    AGE           -0.0500      0.015     -3.242      0.001      -0.080      -0.020
    DIS           -1.3847      0.250     -5.529      0.000      -1.877      -0.892
    RAD            0.2134      0.089      2.390      0.017       0.038       0.389
    TAX           -0.0109      0.005     -2.181      0.030      -0.021      -0.001
    PTRATIO       -1.0141      0.157     -6.474      0.000      -1.322      -0.706
    B              0.0113      0.003      3.403      0.001       0.005       0.018
    ==============================================================================
    Omnibus:                      163.420   Durbin-Watson:                   2.073
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1522.080
    Skew:                           1.688   Prob(JB):                         0.00
    Kurtosis:                      12.581   Cond. No.                     1.47e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.47e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")

plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_19_0.png)
    



```python
predicted = result.predict(X_test)

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.5440575172606662
    mse: 38.50971356874352
    rmse: 6.205619515305746
    mae: 3.6631698981083085


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.731
    Model:                            OLS   Adj. R-squared:                  0.724
    Method:                 Least Squares   F-statistic:                     93.40
    Date:                Tue, 09 Mar 2021   Prob (F-statistic):           1.36e-91
    Time:                        17:40:14   Log-Likelihood:                 45.713
    No. Observations:                 354   AIC:                            -69.43
    Df Residuals:                     343   BIC:                            -26.86
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       2.9167      0.265     11.010      0.000       2.396       3.438
    CRIM           -0.0108      0.002     -6.747      0.000      -0.014      -0.008
    CHAS            0.0904      0.045      2.002      0.046       0.002       0.179
    NOX            -0.9628      0.207     -4.647      0.000      -1.370      -0.555
    RM              0.2548      0.019     13.690      0.000       0.218       0.291
    DIS            -0.0472      0.010     -4.788      0.000      -0.067      -0.028
    RAD             0.0055      0.004      1.396      0.164      -0.002       0.013
    TAX            -0.0003      0.000     -1.386      0.167      -0.001       0.000
    PTRATIO        -0.0413      0.007     -6.284      0.000      -0.054      -0.028
    B               0.0005      0.000      3.638      0.000       0.000       0.001
    pow(AGE, 2) -1.959e-05    5.7e-06     -3.436      0.001   -3.08e-05   -8.38e-06
    ==============================================================================
    Omnibus:                       55.207   Durbin-Watson:                   2.022
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              496.203
    Skew:                           0.228   Prob(JB):                    1.78e-108
    Kurtosis:                       8.782   Cond. No.                     1.74e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.74e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.


Let's plot the QQ-Plot for the residuals


```python
result.pvalues < 0.05
```




    Intercept       True
    CRIM            True
    CHAS            True
    NOX             True
    RM              True
    DIS             True
    RAD            False
    TAX            False
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

    r2 score: 0.6174918674019962
    mse: 32.30731765893707
    rmse: 5.683952643973829
    mae: 3.179319915520512



```python
fig = sm.qqplot(result.resid, fit=True, line="q")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_26_0.png)
    
