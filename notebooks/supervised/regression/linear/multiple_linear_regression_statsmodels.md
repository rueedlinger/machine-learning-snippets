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
    Dep. Variable:                      y   R-squared (uncentered):                   0.965
    Model:                            OLS   Adj. R-squared (uncentered):              0.964
    Method:                 Least Squares   F-statistic:                              722.4
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):                   4.21e-239
    Time:                        13:35:24   Log-Likelihood:                         -1031.0
    No. Observations:                 354   AIC:                                      2088.
    Df Residuals:                     341   BIC:                                      2138.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0908      0.033     -2.724      0.007      -0.156      -0.025
    ZN             0.0421      0.016      2.651      0.008       0.011       0.073
    INDUS          0.0225      0.069      0.327      0.744      -0.113       0.158
    CHAS           1.4213      1.005      1.414      0.158      -0.556       3.398
    NOX           -0.3008      3.790     -0.079      0.937      -7.756       7.155
    RM             5.7536      0.339     16.986      0.000       5.087       6.420
    AGE           -0.0175      0.015     -1.209      0.227      -0.046       0.011
    DIS           -1.0106      0.221     -4.580      0.000      -1.445      -0.577
    RAD            0.1338      0.069      1.928      0.055      -0.003       0.270
    TAX           -0.0110      0.004     -2.704      0.007      -0.019      -0.003
    PTRATIO       -0.3384      0.124     -2.719      0.007      -0.583      -0.094
    B              0.0160      0.003      4.931      0.000       0.010       0.022
    LSTAT         -0.4534      0.056     -8.036      0.000      -0.564      -0.342
    ==============================================================================
    Omnibus:                      162.831   Durbin-Watson:                   2.010
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1409.857
    Skew:                           1.708   Prob(JB):                    7.14e-307
    Kurtosis:                      12.161   Cond. No.                     8.82e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.82e+03. This might indicate that there are
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
    ZN          True
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

    r2 score: 0.6370685861782551
    mse: 36.50735574275049
    rmse: 6.042131721731205
    mae: 3.782854684078256


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.764
    Model:                            OLS   Adj. R-squared:                  0.754
    Method:                 Least Squares   F-statistic:                     84.45
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           5.67e-98
    Time:                        13:35:24   Log-Likelihood:                -1015.7
    No. Observations:                 354   AIC:                             2059.
    Df Residuals:                     340   BIC:                             2113.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         30.4844      5.498      5.545      0.000      19.670      41.299
    CRIM          -0.1058      0.032     -3.298      0.001      -0.169      -0.043
    ZN             0.0416      0.015      2.727      0.007       0.012       0.072
    INDUS          0.0356      0.066      0.538      0.591      -0.094       0.166
    CHAS           1.4713      0.964      1.526      0.128      -0.425       3.367
    NOX          -12.8944      4.286     -3.008      0.003     -21.326      -4.463
    RM             4.0536      0.447      9.075      0.000       3.175       4.932
    AGE           -0.0151      0.014     -1.085      0.278      -0.042       0.012
    DIS           -1.5175      0.231     -6.583      0.000      -1.971      -1.064
    RAD            0.2522      0.070      3.607      0.000       0.115       0.390
    TAX           -0.0137      0.004     -3.485      0.001      -0.021      -0.006
    PTRATIO       -0.8005      0.146     -5.499      0.000      -1.087      -0.514
    B              0.0112      0.003      3.457      0.001       0.005       0.018
    LSTAT         -0.5236      0.056     -9.423      0.000      -0.633      -0.414
    ==============================================================================
    Omnibus:                      135.390   Durbin-Watson:                   2.026
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              699.254
    Skew:                           1.537   Prob(JB):                    1.44e-152
    Kurtosis:                       9.161   Cond. No.                     1.50e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.5e+04. This might indicate that there are
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

    r2 score: 0.6734196509486206
    mse: 32.850793641319136
    rmse: 5.731561187086738
    mae: 3.6635185405700623


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
      <th>473</th>
      <td>4.64689</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.614</td>
      <td>6.980</td>
      <td>67.6</td>
      <td>2.5329</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>374.68</td>
      <td>11.66</td>
      <td>29.8</td>
    </tr>
    <tr>
      <th>168</th>
      <td>2.30040</td>
      <td>0.0</td>
      <td>19.58</td>
      <td>0.0</td>
      <td>0.605</td>
      <td>6.319</td>
      <td>96.1</td>
      <td>2.1000</td>
      <td>5.0</td>
      <td>403.0</td>
      <td>14.7</td>
      <td>297.09</td>
      <td>11.10</td>
      <td>23.8</td>
    </tr>
    <tr>
      <th>346</th>
      <td>0.06162</td>
      <td>0.0</td>
      <td>4.39</td>
      <td>0.0</td>
      <td>0.442</td>
      <td>5.898</td>
      <td>52.3</td>
      <td>8.0136</td>
      <td>3.0</td>
      <td>352.0</td>
      <td>18.8</td>
      <td>364.61</td>
      <td>12.67</td>
      <td>17.2</td>
    </tr>
    <tr>
      <th>444</th>
      <td>12.80230</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>5.854</td>
      <td>96.6</td>
      <td>1.8956</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>240.52</td>
      <td>23.79</td>
      <td>10.8</td>
    </tr>
    <tr>
      <th>406</th>
      <td>20.71620</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.659</td>
      <td>4.138</td>
      <td>100.0</td>
      <td>1.1781</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>370.22</td>
      <td>23.34</td>
      <td>11.9</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.702
    Model:                            OLS   Adj. R-squared:                  0.691
    Method:                 Least Squares   F-statistic:                     66.87
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           5.76e-82
    Time:                        13:35:25   Log-Likelihood:                -1056.7
    No. Observations:                 354   AIC:                             2139.
    Df Residuals:                     341   BIC:                             2190.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     18.6753      6.003      3.111      0.002       6.868      30.483
    CRIM          -0.1411      0.036     -3.951      0.000      -0.211      -0.071
    ZN             0.0320      0.017      1.874      0.062      -0.002       0.065
    INDUS         -0.0210      0.074     -0.285      0.776      -0.166       0.124
    CHAS           2.1203      1.078      1.967      0.050      -0.000       4.241
    NOX          -15.4730      4.797     -3.226      0.001     -24.908      -6.038
    RM             6.0554      0.441     13.743      0.000       5.189       6.922
    AGE           -0.0560      0.015     -3.778      0.000      -0.085      -0.027
    DIS           -1.5282      0.258     -5.912      0.000      -2.037      -1.020
    RAD            0.2407      0.078      3.071      0.002       0.087       0.395
    TAX           -0.0135      0.004     -3.051      0.002      -0.022      -0.005
    PTRATIO       -0.9938      0.162     -6.150      0.000      -1.312      -0.676
    B              0.0149      0.004      4.142      0.000       0.008       0.022
    ==============================================================================
    Omnibus:                      227.080   Durbin-Watson:                   2.153
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3216.313
    Skew:                           2.437   Prob(JB):                         0.00
    Kurtosis:                      16.939   Cond. No.                     1.48e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.48e+04. This might indicate that there are
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

    r2 score: 0.631068322009309
    mse: 37.1109237179211
    rmse: 6.091873580264211
    mae: 3.710534814780751


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.710
    Model:                            OLS   Adj. R-squared:                  0.702
    Method:                 Least Squares   F-statistic:                     84.14
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           4.91e-86
    Time:                        13:35:25   Log-Likelihood:                 40.069
    No. Observations:                 354   AIC:                            -58.14
    Df Residuals:                     343   BIC:                            -15.58
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.1012      0.270     11.486      0.000       2.570       3.632
    CRIM           -0.0113      0.002     -7.080      0.000      -0.014      -0.008
    CHAS            0.0995      0.048      2.063      0.040       0.005       0.194
    NOX            -0.7359      0.211     -3.491      0.001      -1.151      -0.321
    RM              0.2223      0.019     11.604      0.000       0.185       0.260
    DIS            -0.0462      0.010     -4.707      0.000      -0.066      -0.027
    RAD             0.0111      0.003      3.296      0.001       0.004       0.018
    TAX            -0.0006      0.000     -3.662      0.000      -0.001      -0.000
    PTRATIO        -0.0452      0.007     -6.692      0.000      -0.059      -0.032
    B               0.0007      0.000      4.261      0.000       0.000       0.001
    pow(AGE, 2) -2.261e-05   5.61e-06     -4.031      0.000   -3.36e-05   -1.16e-05
    ==============================================================================
    Omnibus:                      106.475   Durbin-Watson:                   2.089
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1056.782
    Skew:                           0.941   Prob(JB):                    3.33e-230
    Kurtosis:                      11.253   Cond. No.                     1.73e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.73e+05. This might indicate that there are
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

    r2 score: 0.6367226995485333
    mse: 36.54214855967682
    rmse: 6.045010219981172
    mae: 3.3644213600940978



```python
fig = sm.qqplot(result.resid, fit=True, line="q")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_26_0.png)
    
