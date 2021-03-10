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
    Dep. Variable:                      y   R-squared (uncentered):                   0.960
    Model:                            OLS   Adj. R-squared (uncentered):              0.958
    Method:                 Least Squares   F-statistic:                              629.8
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):                   2.41e-229
    Time:                        21:26:41   Log-Likelihood:                         -1058.0
    No. Observations:                 354   AIC:                                      2142.
    Df Residuals:                     341   BIC:                                      2192.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0912      0.039     -2.330      0.020      -0.168      -0.014
    ZN             0.0410      0.016      2.494      0.013       0.009       0.073
    INDUS          0.0358      0.075      0.475      0.635      -0.112       0.184
    CHAS           4.2653      1.056      4.039      0.000       2.188       6.342
    NOX           -7.6010      3.867     -1.966      0.050     -15.207       0.005
    RM             6.2655      0.378     16.576      0.000       5.522       7.009
    AGE           -0.0019      0.017     -0.113      0.910      -0.035       0.032
    DIS           -0.9292      0.225     -4.124      0.000      -1.372      -0.486
    RAD            0.1841      0.075      2.462      0.014       0.037       0.331
    TAX           -0.0085      0.004     -1.941      0.053      -0.017       0.000
    PTRATIO       -0.4525      0.128     -3.529      0.000      -0.705      -0.200
    B              0.0154      0.003      4.977      0.000       0.009       0.021
    LSTAT         -0.4284      0.062     -6.888      0.000      -0.551      -0.306
    ==============================================================================
    Omnibus:                      149.048   Durbin-Watson:                   2.016
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              892.242
    Skew:                           1.663   Prob(JB):                    1.79e-194
    Kurtosis:                      10.030   Cond. No.                     8.38e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.38e+03. This might indicate that there are
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

    r2 score: 0.616894860048664
    mse: 27.737820138079986
    rmse: 5.266670688212809
    mae: 3.2794206400754704


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.762
    Model:                            OLS   Adj. R-squared:                  0.753
    Method:                 Least Squares   F-statistic:                     83.66
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           1.91e-97
    Time:                        21:26:41   Log-Likelihood:                -1042.8
    No. Observations:                 354   AIC:                             2114.
    Df Residuals:                     340   BIC:                             2168.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         33.4168      6.043      5.530      0.000      21.531      45.303
    CRIM          -0.1041      0.038     -2.768      0.006      -0.178      -0.030
    ZN             0.0354      0.016      2.240      0.026       0.004       0.067
    INDUS          0.0613      0.072      0.847      0.398      -0.081       0.204
    CHAS           3.9684      1.014      3.912      0.000       1.973       5.964
    NOX          -21.3695      4.467     -4.783      0.000     -30.157     -12.582
    RM             4.4046      0.495      8.904      0.000       3.432       5.378
    AGE            0.0043      0.016      0.265      0.791      -0.028       0.037
    DIS           -1.4012      0.232     -6.030      0.000      -1.858      -0.944
    RAD            0.3027      0.075      4.043      0.000       0.155       0.450
    TAX           -0.0109      0.004     -2.580      0.010      -0.019      -0.003
    PTRATIO       -0.9997      0.158     -6.333      0.000      -1.310      -0.689
    B              0.0106      0.003      3.422      0.001       0.004       0.017
    LSTAT         -0.5183      0.062     -8.382      0.000      -0.640      -0.397
    ==============================================================================
    Omnibus:                      127.414   Durbin-Watson:                   2.007
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              520.363
    Skew:                           1.524   Prob(JB):                    1.01e-113
    Kurtosis:                       8.097   Cond. No.                     1.52e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.52e+04. This might indicate that there are
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

    r2 score: 0.6592358722069794
    mse: 24.67221945242801
    rmse: 4.9671137949948365
    mae: 3.3529304031561513


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
      <th>115</th>
      <td>0.17134</td>
      <td>0.0</td>
      <td>10.01</td>
      <td>0.0</td>
      <td>0.547</td>
      <td>5.928</td>
      <td>88.2</td>
      <td>2.4631</td>
      <td>6.0</td>
      <td>432.0</td>
      <td>17.8</td>
      <td>344.91</td>
      <td>15.76</td>
      <td>18.3</td>
    </tr>
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
      <th>311</th>
      <td>0.79041</td>
      <td>0.0</td>
      <td>9.90</td>
      <td>0.0</td>
      <td>0.544</td>
      <td>6.122</td>
      <td>52.8</td>
      <td>2.6403</td>
      <td>4.0</td>
      <td>304.0</td>
      <td>18.4</td>
      <td>396.90</td>
      <td>5.98</td>
      <td>22.1</td>
    </tr>
    <tr>
      <th>257</th>
      <td>0.61154</td>
      <td>20.0</td>
      <td>3.97</td>
      <td>0.0</td>
      <td>0.647</td>
      <td>8.704</td>
      <td>86.9</td>
      <td>1.8010</td>
      <td>5.0</td>
      <td>264.0</td>
      <td>13.0</td>
      <td>389.70</td>
      <td>5.12</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>0.22212</td>
      <td>0.0</td>
      <td>10.01</td>
      <td>0.0</td>
      <td>0.547</td>
      <td>6.092</td>
      <td>95.4</td>
      <td>2.5480</td>
      <td>6.0</td>
      <td>432.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>17.09</td>
      <td>18.7</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.713
    Model:                            OLS   Adj. R-squared:                  0.702
    Method:                 Least Squares   F-statistic:                     70.46
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           1.14e-84
    Time:                        21:26:41   Log-Likelihood:                -1076.0
    No. Observations:                 354   AIC:                             2178.
    Df Residuals:                     341   BIC:                             2228.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     20.0978      6.395      3.143      0.002       7.519      32.676
    CRIM          -0.1500      0.041     -3.673      0.000      -0.230      -0.070
    ZN             0.0269      0.017      1.553      0.121      -0.007       0.061
    INDUS          0.0026      0.079      0.032      0.974      -0.153       0.158
    CHAS           4.6650      1.109      4.207      0.000       2.484       6.846
    NOX          -24.4038      4.884     -4.997      0.000     -34.010     -14.797
    RM             6.5982      0.460     14.330      0.000       5.693       7.504
    AGE           -0.0488      0.017     -2.953      0.003      -0.081      -0.016
    DIS           -1.5695      0.254     -6.181      0.000      -2.069      -1.070
    RAD            0.2788      0.082      3.398      0.001       0.117       0.440
    TAX           -0.0106      0.005     -2.294      0.022      -0.020      -0.002
    PTRATIO       -1.0683      0.173     -6.178      0.000      -1.408      -0.728
    B              0.0138      0.003      4.118      0.000       0.007       0.020
    ==============================================================================
    Omnibus:                      204.355   Durbin-Watson:                   2.014
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2162.357
    Skew:                           2.215   Prob(JB):                         0.00
    Kurtosis:                      14.268   Cond. No.                     1.50e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.5e+04. This might indicate that there are
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

    r2 score: 0.5807267918731706
    mse: 30.356483437458024
    rmse: 5.509671808507111
    mae: 3.579089684187565


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.728
    Model:                            OLS   Adj. R-squared:                  0.720
    Method:                 Least Squares   F-statistic:                     91.92
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           9.85e-91
    Time:                        21:26:42   Log-Likelihood:                 38.542
    No. Observations:                 354   AIC:                            -55.08
    Df Residuals:                     343   BIC:                            -12.52
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.1314      0.273     11.477      0.000       2.595       3.668
    CRIM           -0.0126      0.002     -7.255      0.000      -0.016      -0.009
    CHAS            0.1759      0.047      3.723      0.000       0.083       0.269
    NOX            -1.0197      0.202     -5.047      0.000      -1.417      -0.622
    RM              0.2333      0.019     12.180      0.000       0.196       0.271
    DIS            -0.0507      0.010     -5.237      0.000      -0.070      -0.032
    RAD             0.0129      0.003      3.802      0.000       0.006       0.020
    TAX            -0.0005      0.000     -2.985      0.003      -0.001      -0.000
    PTRATIO        -0.0446      0.007     -6.483      0.000      -0.058      -0.031
    B               0.0007      0.000      4.782      0.000       0.000       0.001
    pow(AGE, 2) -2.058e-05   5.93e-06     -3.470      0.001   -3.22e-05   -8.91e-06
    ==============================================================================
    Omnibus:                      100.348   Durbin-Watson:                   1.969
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              797.932
    Skew:                           0.940   Prob(JB):                    5.39e-174
    Kurtosis:                      10.111   Cond. No.                     1.74e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.74e+05. This might indicate that there are
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

    r2 score: 0.6246027759735415
    mse: 27.179746744465756
    rmse: 5.213419870340942
    mae: 3.291826842536456



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
