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
    Method:                 Least Squares   F-statistic:                              568.7
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):                   4.13e-222
    Time:                        14:32:43   Log-Likelihood:                         -1087.9
    No. Observations:                 354   AIC:                                      2202.
    Df Residuals:                     341   BIC:                                      2252.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1463      0.044     -3.348      0.001      -0.232      -0.060
    ZN             0.0607      0.018      3.415      0.001       0.026       0.096
    INDUS          0.0157      0.085      0.185      0.853      -0.151       0.182
    CHAS           2.6276      1.126      2.334      0.020       0.414       4.842
    NOX           -3.2278      4.306     -0.750      0.454     -11.698       5.242
    RM             5.8257      0.386     15.096      0.000       5.067       6.585
    AGE            0.0037      0.018      0.205      0.837      -0.032       0.039
    DIS           -1.0507      0.238     -4.417      0.000      -1.519      -0.583
    RAD            0.2107      0.088      2.406      0.017       0.038       0.383
    TAX           -0.0089      0.005     -1.712      0.088      -0.019       0.001
    PTRATIO       -0.3582      0.135     -2.660      0.008      -0.623      -0.093
    B              0.0150      0.003      4.297      0.000       0.008       0.022
    LSTAT         -0.4741      0.064     -7.360      0.000      -0.601      -0.347
    ==============================================================================
    Omnibus:                      128.181   Durbin-Watson:                   1.942
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              708.384
    Skew:                           1.416   Prob(JB):                    1.50e-154
    Kurtosis:                       9.325   Cond. No.                     8.51e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.51e+03. This might indicate that there are
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

    r2 score: 0.7270752435956049
    mse: 17.96348019470387
    rmse: 4.238334601550929
    mae: 3.113353648895026


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.730
    Model:                            OLS   Adj. R-squared:                  0.720
    Method:                 Least Squares   F-statistic:                     70.70
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           2.83e-88
    Time:                        14:32:43   Log-Likelihood:                -1070.4
    No. Observations:                 354   AIC:                             2169.
    Df Residuals:                     340   BIC:                             2223.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         37.4544      6.312      5.934      0.000      25.039      49.870
    CRIM          -0.1533      0.042     -3.676      0.000      -0.235      -0.071
    ZN             0.0565      0.017      3.332      0.001       0.023       0.090
    INDUS          0.0228      0.081      0.283      0.777      -0.136       0.181
    CHAS           2.6410      1.073      2.461      0.014       0.530       4.752
    NOX          -18.4109      4.837     -3.806      0.000     -27.926      -8.896
    RM             3.7158      0.512      7.263      0.000       2.709       4.722
    AGE            0.0061      0.017      0.358      0.720      -0.027       0.040
    DIS           -1.5857      0.244     -6.498      0.000      -2.066      -1.106
    RAD            0.3395      0.086      3.936      0.000       0.170       0.509
    TAX           -0.0122      0.005     -2.437      0.015      -0.022      -0.002
    PTRATIO       -0.9295      0.160     -5.793      0.000      -1.245      -0.614
    B              0.0093      0.003      2.702      0.007       0.003       0.016
    LSTAT         -0.5637      0.063     -8.914      0.000      -0.688      -0.439
    ==============================================================================
    Omnibus:                      115.327   Durbin-Watson:                   1.945
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              416.250
    Skew:                           1.412   Prob(JB):                     4.10e-91
    Kurtosis:                       7.500   Cond. No.                     1.46e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.46e+04. This might indicate that there are
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

    r2 score: 0.7561499765770228
    mse: 16.04982678722723
    rmse: 4.006223506898639
    mae: 3.031027353213903


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
      <th>315</th>
      <td>0.25356</td>
      <td>0.0</td>
      <td>9.90</td>
      <td>0.0</td>
      <td>0.544</td>
      <td>5.705</td>
      <td>77.7</td>
      <td>3.9450</td>
      <td>4.0</td>
      <td>304.0</td>
      <td>18.4</td>
      <td>396.42</td>
      <td>11.50</td>
      <td>16.2</td>
    </tr>
    <tr>
      <th>501</th>
      <td>0.06263</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.593</td>
      <td>69.1</td>
      <td>2.4786</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>391.99</td>
      <td>9.67</td>
      <td>22.4</td>
    </tr>
    <tr>
      <th>381</th>
      <td>15.87440</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.671</td>
      <td>6.545</td>
      <td>99.1</td>
      <td>1.5192</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>21.08</td>
      <td>10.9</td>
    </tr>
    <tr>
      <th>388</th>
      <td>14.33370</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.700</td>
      <td>4.880</td>
      <td>100.0</td>
      <td>1.5895</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>372.92</td>
      <td>30.62</td>
      <td>10.2</td>
    </tr>
    <tr>
      <th>422</th>
      <td>12.04820</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.614</td>
      <td>5.648</td>
      <td>87.6</td>
      <td>1.9512</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>291.55</td>
      <td>14.10</td>
      <td>20.8</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.667
    Model:                            OLS   Adj. R-squared:                  0.655
    Method:                 Least Squares   F-statistic:                     56.88
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           7.17e-74
    Time:                        14:32:44   Log-Likelihood:                -1107.6
    No. Observations:                 354   AIC:                             2241.
    Df Residuals:                     341   BIC:                             2291.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     24.0217      6.798      3.534      0.000      10.650      37.394
    CRIM          -0.1918      0.046     -4.171      0.000      -0.282      -0.101
    ZN             0.0507      0.019      2.698      0.007       0.014       0.088
    INDUS         -0.0069      0.089     -0.077      0.939      -0.183       0.169
    CHAS           3.1433      1.189      2.645      0.009       0.806       5.481
    NOX          -24.6549      5.309     -4.644      0.000     -35.096     -14.213
    RM             6.0458      0.488     12.394      0.000       5.086       7.005
    AGE           -0.0430      0.018     -2.398      0.017      -0.078      -0.008
    DIS           -1.7355      0.270     -6.428      0.000      -2.267      -1.204
    RAD            0.3543      0.096      3.704      0.000       0.166       0.542
    TAX           -0.0133      0.006     -2.392      0.017      -0.024      -0.002
    PTRATIO       -1.0299      0.178     -5.802      0.000      -1.379      -0.681
    B              0.0141      0.004      3.708      0.000       0.007       0.022
    ==============================================================================
    Omnibus:                      192.072   Durbin-Watson:                   1.925
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1543.976
    Skew:                           2.150   Prob(JB):                         0.00
    Kurtosis:                      12.283   Cond. No.                     1.45e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.45e+04. This might indicate that there are
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

    r2 score: 0.7212201388029327
    mse: 18.34885402581665
    rmse: 4.283556235864851
    mae: 3.3208512078168564


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.682
    Model:                            OLS   Adj. R-squared:                  0.673
    Method:                 Least Squares   F-statistic:                     73.62
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           3.55e-79
    Time:                        14:32:44   Log-Likelihood:                 16.483
    No. Observations:                 354   AIC:                            -10.97
    Df Residuals:                     343   BIC:                             31.60
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.3443      0.283     11.828      0.000       2.788       3.900
    CRIM           -0.0128      0.002     -6.723      0.000      -0.017      -0.009
    CHAS            0.1242      0.049      2.517      0.012       0.027       0.221
    NOX            -1.1103      0.215     -5.152      0.000      -1.534      -0.686
    RM              0.2167      0.020     10.954      0.000       0.178       0.256
    DIS            -0.0508      0.010     -5.199      0.000      -0.070      -0.032
    RAD             0.0150      0.004      3.893      0.000       0.007       0.023
    TAX            -0.0006      0.000     -2.810      0.005      -0.001      -0.000
    PTRATIO        -0.0461      0.007     -6.598      0.000      -0.060      -0.032
    B               0.0006      0.000      4.009      0.000       0.000       0.001
    pow(AGE, 2) -1.778e-05   6.06e-06     -2.935      0.004   -2.97e-05   -5.86e-06
    ==============================================================================
    Omnibus:                      102.421   Durbin-Watson:                   2.014
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              648.443
    Skew:                           1.042   Prob(JB):                    1.56e-141
    Kurtosis:                       9.294   Cond. No.                     1.68e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.68e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.


Let's plot the QQ-Plot for the residuals


```python
result.pvalues < 0.05
```




    Intercept      True
    CRIM           True
    CHAS           True
    NOX            True
    RM             True
    DIS            True
    RAD            True
    TAX            True
    PTRATIO        True
    B              True
    pow(AGE, 2)    True
    dtype: bool




```python
predicted = np.exp(result.predict(X_test))

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.8023289698068206
    mse: 13.010397747430979
    rmse: 3.606992895395135
    mae: 2.771814467169883



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
