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
    Dep. Variable:                      y   R-squared (uncentered):                   0.959
    Model:                            OLS   Adj. R-squared (uncentered):              0.958
    Method:                 Least Squares   F-statistic:                              621.3
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):                   2.24e-228
    Time:                        17:50:42   Log-Likelihood:                         -1059.0
    No. Observations:                 354   AIC:                                      2144.
    Df Residuals:                     341   BIC:                                      2194.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0774      0.065     -1.198      0.232      -0.204       0.050
    ZN             0.0481      0.016      3.004      0.003       0.017       0.080
    INDUS          0.0470      0.076      0.618      0.537      -0.102       0.196
    CHAS           1.0791      1.011      1.068      0.286      -0.909       3.067
    NOX           -4.8429      3.932     -1.232      0.219     -12.576       2.890
    RM             5.9443      0.379     15.690      0.000       5.199       6.690
    AGE           -0.0060      0.016     -0.384      0.701      -0.037       0.025
    DIS           -0.9327      0.221     -4.214      0.000      -1.368      -0.497
    RAD            0.1133      0.081      1.397      0.163      -0.046       0.273
    TAX           -0.0079      0.005     -1.697      0.091      -0.017       0.001
    PTRATIO       -0.3401      0.130     -2.616      0.009      -0.596      -0.084
    B              0.0127      0.003      3.803      0.000       0.006       0.019
    LSTAT         -0.4160      0.060     -6.984      0.000      -0.533      -0.299
    ==============================================================================
    Omnibus:                      193.163   Durbin-Watson:                   1.914
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1921.442
    Skew:                           2.077   Prob(JB):                         0.00
    Kurtosis:                      13.631   Cond. No.                     8.43e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.43e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_8_0.png)
    



```python
result.pvalues < 0.05
```




    CRIM       False
    ZN          True
    INDUS      False
    CHAS       False
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

    r2 score: 0.7496703171701523
    mse: 27.683398701036506
    rmse: 5.261501563340688
    mae: 3.715267710128775


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.705
    Model:                            OLS   Adj. R-squared:                  0.693
    Method:                 Least Squares   F-statistic:                     62.40
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           9.52e-82
    Time:                        17:50:42   Log-Likelihood:                -1046.1
    No. Observations:                 354   AIC:                             2120.
    Df Residuals:                     340   BIC:                             2174.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         31.5267      6.213      5.074      0.000      19.305      43.748
    CRIM          -0.1205      0.063     -1.916      0.056      -0.244       0.003
    ZN             0.0473      0.015      3.059      0.002       0.017       0.078
    INDUS          0.0597      0.073      0.813      0.417      -0.085       0.204
    CHAS           0.9872      0.976      1.011      0.313      -0.933       2.907
    NOX          -16.4812      4.435     -3.716      0.000     -25.205      -7.757
    RM             4.0348      0.525      7.688      0.000       3.002       5.067
    AGE           -0.0041      0.015     -0.271      0.787      -0.034       0.026
    DIS           -1.3905      0.232     -5.994      0.000      -1.847      -0.934
    RAD            0.2459      0.083      2.978      0.003       0.083       0.408
    TAX           -0.0108      0.005     -2.375      0.018      -0.020      -0.002
    PTRATIO       -0.8147      0.157     -5.204      0.000      -1.123      -0.507
    B              0.0079      0.003      2.367      0.019       0.001       0.015
    LSTAT         -0.5023      0.060     -8.374      0.000      -0.620      -0.384
    ==============================================================================
    Omnibus:                      176.318   Durbin-Watson:                   1.953
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1210.912
    Skew:                           1.988   Prob(JB):                    1.13e-263
    Kurtosis:                      11.142   Cond. No.                     1.51e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.51e+04. This might indicate that there are
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

    r2 score: 0.7831435188657527
    mse: 23.981672330179496
    rmse: 4.8971085683471935
    mae: 3.4576855853797674


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
      <th>146</th>
      <td>2.15505</td>
      <td>0.0</td>
      <td>19.58</td>
      <td>0.0</td>
      <td>0.871</td>
      <td>5.628</td>
      <td>100.0</td>
      <td>1.5166</td>
      <td>5.0</td>
      <td>403.0</td>
      <td>14.7</td>
      <td>169.27</td>
      <td>16.65</td>
      <td>15.6</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.25179</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.570</td>
      <td>98.1</td>
      <td>3.7979</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>376.57</td>
      <td>21.02</td>
      <td>13.6</td>
    </tr>
    <tr>
      <th>255</th>
      <td>0.03548</td>
      <td>80.0</td>
      <td>3.64</td>
      <td>0.0</td>
      <td>0.392</td>
      <td>5.876</td>
      <td>19.1</td>
      <td>9.2203</td>
      <td>1.0</td>
      <td>315.0</td>
      <td>16.4</td>
      <td>395.18</td>
      <td>9.25</td>
      <td>20.9</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.25387</td>
      <td>0.0</td>
      <td>6.91</td>
      <td>0.0</td>
      <td>0.448</td>
      <td>5.399</td>
      <td>95.3</td>
      <td>5.8700</td>
      <td>3.0</td>
      <td>233.0</td>
      <td>17.9</td>
      <td>396.90</td>
      <td>30.81</td>
      <td>14.4</td>
    </tr>
    <tr>
      <th>329</th>
      <td>0.06724</td>
      <td>0.0</td>
      <td>3.24</td>
      <td>0.0</td>
      <td>0.460</td>
      <td>6.333</td>
      <td>17.2</td>
      <td>5.2146</td>
      <td>4.0</td>
      <td>430.0</td>
      <td>16.9</td>
      <td>375.21</td>
      <td>7.34</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.644
    Model:                            OLS   Adj. R-squared:                  0.631
    Method:                 Least Squares   F-statistic:                     51.35
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           5.54e-69
    Time:                        17:50:42   Log-Likelihood:                -1079.2
    No. Observations:                 354   AIC:                             2184.
    Df Residuals:                     341   BIC:                             2235.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     16.7793      6.535      2.568      0.011       3.926      29.633
    CRIM          -0.1985      0.068     -2.909      0.004      -0.333      -0.064
    ZN             0.0427      0.017      2.518      0.012       0.009       0.076
    INDUS          0.0226      0.080      0.281      0.779      -0.135       0.181
    CHAS           1.1004      1.070      1.028      0.305      -1.005       3.206
    NOX          -18.4514      4.857     -3.799      0.000     -28.005      -8.897
    RM             6.2694      0.496     12.649      0.000       5.294       7.244
    AGE           -0.0492      0.016     -3.160      0.002      -0.080      -0.019
    DIS           -1.4884      0.254     -5.858      0.000      -1.988      -0.989
    RAD            0.2450      0.091      2.706      0.007       0.067       0.423
    TAX           -0.0110      0.005     -2.207      0.028      -0.021      -0.001
    PTRATIO       -0.8980      0.171     -5.241      0.000      -1.235      -0.561
    B              0.0115      0.004      3.163      0.002       0.004       0.019
    ==============================================================================
    Omnibus:                      233.068   Durbin-Watson:                   2.054
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3160.291
    Skew:                           2.548   Prob(JB):                         0.00
    Kurtosis:                      16.722   Cond. No.                     1.48e+04
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

    r2 score: 0.7300025942201465
    mse: 29.858408111872617
    rmse: 5.4642847758762185
    mae: 3.6904444809675634


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.663
    Model:                            OLS   Adj. R-squared:                  0.653
    Method:                 Least Squares   F-statistic:                     67.44
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           7.75e-75
    Time:                        17:50:42   Log-Likelihood:                 33.875
    No. Observations:                 354   AIC:                            -45.75
    Df Residuals:                     343   BIC:                            -3.188
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.1135      0.280     11.130      0.000       2.563       3.664
    CRIM           -0.0172      0.003     -5.888      0.000      -0.023      -0.011
    CHAS            0.0673      0.046      1.467      0.143      -0.023       0.158
    NOX            -0.8615      0.205     -4.200      0.000      -1.265      -0.458
    RM              0.2259      0.021     10.938      0.000       0.185       0.267
    DIS            -0.0495      0.010     -5.153      0.000      -0.068      -0.031
    RAD             0.0110      0.004      2.946      0.003       0.004       0.018
    TAX            -0.0004      0.000     -2.252      0.025      -0.001   -5.44e-05
    PTRATIO        -0.0405      0.007     -5.783      0.000      -0.054      -0.027
    B               0.0004      0.000      2.836      0.005       0.000       0.001
    pow(AGE, 2) -2.121e-05    5.6e-06     -3.789      0.000   -3.22e-05   -1.02e-05
    ==============================================================================
    Omnibus:                      118.848   Durbin-Watson:                   2.155
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              894.243
    Skew:                           1.188   Prob(JB):                    6.57e-195
    Kurtosis:                      10.415   Cond. No.                     1.72e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.72e+05. This might indicate that there are
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

    r2 score: 0.7753376923840487
    mse: 24.844901189982757
    rmse: 4.984465988446782
    mae: 3.0912974099283534



```python
fig = sm.qqplot(result.resid, fit=True, line="q")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_26_0.png)
    
