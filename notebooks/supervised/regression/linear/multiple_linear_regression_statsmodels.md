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
    Dep. Variable:                      y   R-squared (uncentered):                   0.957
    Model:                            OLS   Adj. R-squared (uncentered):              0.955
    Method:                 Least Squares   F-statistic:                              580.5
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):                   1.46e-223
    Time:                        22:23:26   Log-Likelihood:                         -1078.5
    No. Observations:                 354   AIC:                                      2183.
    Df Residuals:                     341   BIC:                                      2233.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1079      0.043     -2.525      0.012      -0.192      -0.024
    ZN             0.0578      0.018      3.210      0.001       0.022       0.093
    INDUS          0.0373      0.080      0.467      0.641      -0.120       0.195
    CHAS           2.9164      1.103      2.644      0.009       0.746       5.086
    NOX           -4.7260      4.357     -1.085      0.279     -13.295       3.843
    RM             6.0198      0.391     15.407      0.000       5.251       6.788
    AGE           -0.0120      0.018     -0.676      0.499      -0.047       0.023
    DIS           -1.1363      0.250     -4.543      0.000      -1.628      -0.644
    RAD            0.1139      0.083      1.378      0.169      -0.049       0.276
    TAX           -0.0078      0.005     -1.633      0.103      -0.017       0.002
    PTRATIO       -0.3302      0.142     -2.324      0.021      -0.610      -0.051
    B              0.0141      0.003      4.127      0.000       0.007       0.021
    LSTAT         -0.4171      0.064     -6.516      0.000      -0.543      -0.291
    ==============================================================================
    Omnibus:                      160.480   Durbin-Watson:                   1.970
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1164.827
    Skew:                           1.743   Prob(JB):                    1.15e-253
    Kurtosis:                      11.174   Cond. No.                     8.92e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.92e+03. This might indicate that there are
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
    CHAS        True
    NOX        False
    RM          True
    AGE        False
    DIS         True
    RAD        False
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

    r2 score: 0.6957989525248272
    mse: 20.61608207871863
    rmse: 4.540493594172183
    mae: 3.167188306893607


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.737
    Model:                            OLS   Adj. R-squared:                  0.727
    Method:                 Least Squares   F-statistic:                     73.16
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           4.14e-90
    Time:                        22:23:26   Log-Likelihood:                -1065.7
    No. Observations:                 354   AIC:                             2159.
    Df Residuals:                     340   BIC:                             2213.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         33.9228      6.718      5.050      0.000      20.709      47.136
    CRIM          -0.1072      0.041     -2.598      0.010      -0.188      -0.026
    ZN             0.0570      0.017      3.279      0.001       0.023       0.091
    INDUS          0.0471      0.077      0.610      0.542      -0.105       0.199
    CHAS           2.7161      1.066      2.547      0.011       0.619       4.814
    NOX          -17.9832      4.960     -3.626      0.000     -27.739      -8.227
    RM             3.9730      0.554      7.174      0.000       2.884       5.062
    AGE           -0.0063      0.017     -0.365      0.715      -0.040       0.028
    DIS           -1.6212      0.260     -6.236      0.000      -2.133      -1.110
    RAD            0.2292      0.083      2.762      0.006       0.066       0.393
    TAX           -0.0103      0.005     -2.224      0.027      -0.019      -0.001
    PTRATIO       -0.8112      0.167     -4.856      0.000      -1.140      -0.483
    B              0.0083      0.003      2.387      0.018       0.001       0.015
    LSTAT         -0.5271      0.066     -8.041      0.000      -0.656      -0.398
    ==============================================================================
    Omnibus:                      145.214   Durbin-Watson:                   2.058
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              693.023
    Skew:                           1.703   Prob(JB):                    3.25e-151
    Kurtosis:                       8.948   Cond. No.                     1.58e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.58e+04. This might indicate that there are
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

    r2 score: 0.7444514188387512
    mse: 17.318844126433756
    rmse: 4.161591537673268
    mae: 3.0381303526279457


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
      <th>117</th>
      <td>0.15098</td>
      <td>0.0</td>
      <td>10.01</td>
      <td>0.0</td>
      <td>0.547</td>
      <td>6.021</td>
      <td>82.6</td>
      <td>2.7474</td>
      <td>6.0</td>
      <td>432.0</td>
      <td>17.8</td>
      <td>394.51</td>
      <td>10.30</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>377</th>
      <td>9.82349</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.671</td>
      <td>6.794</td>
      <td>98.8</td>
      <td>1.3580</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>21.24</td>
      <td>13.3</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.84054</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.599</td>
      <td>85.7</td>
      <td>4.4546</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>303.42</td>
      <td>16.51</td>
      <td>13.9</td>
    </tr>
    <tr>
      <th>275</th>
      <td>0.09604</td>
      <td>40.0</td>
      <td>6.41</td>
      <td>0.0</td>
      <td>0.447</td>
      <td>6.854</td>
      <td>42.8</td>
      <td>4.2673</td>
      <td>4.0</td>
      <td>254.0</td>
      <td>17.6</td>
      <td>396.90</td>
      <td>2.98</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>205</th>
      <td>0.13642</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>0.0</td>
      <td>0.489</td>
      <td>5.891</td>
      <td>22.3</td>
      <td>3.9454</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>396.90</td>
      <td>10.87</td>
      <td>22.6</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.687
    Model:                            OLS   Adj. R-squared:                  0.676
    Method:                 Least Squares   F-statistic:                     62.25
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           2.49e-78
    Time:                        22:23:26   Log-Likelihood:                -1096.5
    No. Observations:                 354   AIC:                             2219.
    Df Residuals:                     341   BIC:                             2269.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     15.9696      6.902      2.314      0.021       2.394      29.545
    CRIM          -0.1750      0.044     -3.977      0.000      -0.262      -0.088
    ZN             0.0477      0.019      2.525      0.012       0.011       0.085
    INDUS          0.0338      0.084      0.401      0.688      -0.132       0.199
    CHAS           3.3155      1.159      2.861      0.004       1.036       5.595
    NOX          -22.7242      5.365     -4.236      0.000     -33.277     -12.172
    RM             6.5752      0.490     13.430      0.000       5.612       7.538
    AGE           -0.0518      0.018     -2.928      0.004      -0.087      -0.017
    DIS           -1.6743      0.283     -5.914      0.000      -2.231      -1.117
    RAD            0.2167      0.090      2.397      0.017       0.039       0.395
    TAX           -0.0092      0.005     -1.820      0.070      -0.019       0.001
    PTRATIO       -0.8965      0.182     -4.937      0.000      -1.254      -0.539
    B              0.0151      0.004      4.095      0.000       0.008       0.022
    ==============================================================================
    Omnibus:                      207.077   Durbin-Watson:                   2.018
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2149.825
    Skew:                           2.264   Prob(JB):                         0.00
    Kurtosis:                      14.191   Cond. No.                     1.54e+04
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

    r2 score: 0.660914968702985
    mse: 22.98021290493639
    rmse: 4.793768132162463
    mae: 3.2998304542692183


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.732
    Model:                            OLS   Adj. R-squared:                  0.724
    Method:                 Least Squares   F-statistic:                     93.53
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           1.14e-91
    Time:                        22:23:27   Log-Likelihood:                 38.243
    No. Observations:                 354   AIC:                            -54.49
    Df Residuals:                     343   BIC:                            -11.92
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       2.9621      0.279     10.624      0.000       2.414       3.510
    CRIM           -0.0140      0.002     -7.896      0.000      -0.017      -0.010
    CHAS            0.1382      0.047      2.959      0.003       0.046       0.230
    NOX            -1.0404      0.212     -4.898      0.000      -1.458      -0.623
    RM              0.2436      0.019     12.731      0.000       0.206       0.281
    DIS            -0.0518      0.010     -5.385      0.000      -0.071      -0.033
    RAD             0.0084      0.004      2.377      0.018       0.001       0.015
    TAX            -0.0003      0.000     -1.516      0.130      -0.001    8.15e-05
    PTRATIO        -0.0418      0.007     -6.132      0.000      -0.055      -0.028
    B               0.0008      0.000      5.055      0.000       0.000       0.001
    pow(AGE, 2) -2.086e-05   5.77e-06     -3.613      0.000   -3.22e-05    -9.5e-06
    ==============================================================================
    Omnibus:                      115.738   Durbin-Watson:                   2.045
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              858.752
    Skew:                           1.154   Prob(JB):                    3.34e-187
    Kurtosis:                      10.273   Cond. No.                     1.80e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.8e+05. This might indicate that there are
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
    RAD             True
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

    r2 score: 0.7705318695692475
    mse: 15.551339650783422
    rmse: 3.9435186890369143
    mae: 2.8799079642279746



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
