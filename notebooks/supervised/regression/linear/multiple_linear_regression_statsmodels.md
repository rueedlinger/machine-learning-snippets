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
    Dep. Variable:                      y   R-squared (uncentered):                   0.962
    Model:                            OLS   Adj. R-squared (uncentered):              0.961
    Method:                 Least Squares   F-statistic:                              670.7
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):                   8.13e-234
    Time:                        23:13:03   Log-Likelihood:                         -1052.3
    No. Observations:                 354   AIC:                                      2131.
    Df Residuals:                     341   BIC:                                      2181.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1020      0.035     -2.886      0.004      -0.171      -0.032
    ZN             0.0483      0.016      3.030      0.003       0.017       0.080
    INDUS          0.0379      0.074      0.510      0.610      -0.108       0.184
    CHAS           4.9553      1.107      4.477      0.000       2.778       7.133
    NOX           -4.7636      3.859     -1.234      0.218     -12.354       2.827
    RM             6.0664      0.349     17.360      0.000       5.379       6.754
    AGE           -0.0203      0.016     -1.299      0.195      -0.051       0.010
    DIS           -1.0464      0.224     -4.675      0.000      -1.487      -0.606
    RAD            0.1917      0.077      2.495      0.013       0.041       0.343
    TAX           -0.0082      0.005     -1.818      0.070      -0.017       0.001
    PTRATIO       -0.4271      0.128     -3.346      0.001      -0.678      -0.176
    B              0.0161      0.003      5.080      0.000       0.010       0.022
    LSTAT         -0.3791      0.058     -6.501      0.000      -0.494      -0.264
    ==============================================================================
    Omnibus:                      127.346   Durbin-Watson:                   1.938
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              547.339
    Skew:                           1.502   Prob(JB):                    1.40e-119
    Kurtosis:                       8.299   Cond. No.                     8.48e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.48e+03. This might indicate that there are
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

    r2 score: 0.6693456635999928
    mse: 29.971267993447753
    rmse: 5.474602085398331
    mae: 3.630912236080549


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.744
    Model:                            OLS   Adj. R-squared:                  0.734
    Method:                 Least Squares   F-statistic:                     76.01
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           3.58e-92
    Time:                        23:13:04   Log-Likelihood:                -1040.5
    No. Observations:                 354   AIC:                             2109.
    Df Residuals:                     340   BIC:                             2163.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         29.0306      5.984      4.851      0.000      17.260      40.801
    CRIM          -0.1106      0.034     -3.227      0.001      -0.178      -0.043
    ZN             0.0453      0.015      2.932      0.004       0.015       0.076
    INDUS          0.0612      0.072      0.849      0.396      -0.081       0.203
    CHAS           4.7932      1.073      4.469      0.000       2.683       6.903
    NOX          -16.0843      4.406     -3.650      0.000     -24.751      -7.418
    RM             4.3508      0.489      8.888      0.000       3.388       5.314
    AGE           -0.0133      0.015     -0.874      0.383      -0.043       0.017
    DIS           -1.3757      0.227     -6.056      0.000      -1.823      -0.929
    RAD            0.3028      0.078      3.889      0.000       0.150       0.456
    TAX           -0.0108      0.004     -2.457      0.015      -0.020      -0.002
    PTRATIO       -0.8820      0.155     -5.685      0.000      -1.187      -0.577
    B              0.0110      0.003      3.404      0.001       0.005       0.017
    LSTAT         -0.4701      0.060     -7.899      0.000      -0.587      -0.353
    ==============================================================================
    Omnibus:                      115.982   Durbin-Watson:                   1.859
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              399.971
    Skew:                           1.440   Prob(JB):                     1.40e-87
    Kurtosis:                       7.338   Cond. No.                     1.50e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.5e+04. This might indicate that there are
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

    r2 score: 0.7132787903993958
    mse: 25.989068541806148
    rmse: 5.097947483233439
    mae: 3.5059034211161153


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
      <th>483</th>
      <td>2.81838</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.532</td>
      <td>5.762</td>
      <td>40.3</td>
      <td>4.0983</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>392.92</td>
      <td>10.42</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>336</th>
      <td>0.03427</td>
      <td>0.0</td>
      <td>5.19</td>
      <td>0.0</td>
      <td>0.515</td>
      <td>5.869</td>
      <td>46.3</td>
      <td>5.2311</td>
      <td>5.0</td>
      <td>224.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>9.80</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>443</th>
      <td>9.96654</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>6.485</td>
      <td>100.0</td>
      <td>1.9784</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>386.73</td>
      <td>18.85</td>
      <td>15.4</td>
    </tr>
    <tr>
      <th>288</th>
      <td>0.04590</td>
      <td>52.5</td>
      <td>5.32</td>
      <td>0.0</td>
      <td>0.405</td>
      <td>6.315</td>
      <td>45.6</td>
      <td>7.3172</td>
      <td>6.0</td>
      <td>293.0</td>
      <td>16.6</td>
      <td>396.90</td>
      <td>7.60</td>
      <td>22.3</td>
    </tr>
    <tr>
      <th>348</th>
      <td>0.01501</td>
      <td>80.0</td>
      <td>2.01</td>
      <td>0.0</td>
      <td>0.435</td>
      <td>6.635</td>
      <td>29.7</td>
      <td>8.3440</td>
      <td>4.0</td>
      <td>280.0</td>
      <td>17.0</td>
      <td>390.94</td>
      <td>5.99</td>
      <td>24.5</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.697
    Model:                            OLS   Adj. R-squared:                  0.686
    Method:                 Least Squares   F-statistic:                     65.37
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           8.34e-81
    Time:                        23:13:04   Log-Likelihood:                -1070.3
    No. Observations:                 354   AIC:                             2167.
    Df Residuals:                     341   BIC:                             2217.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     14.1344      6.169      2.291      0.023       2.000      26.269
    CRIM          -0.1515      0.037     -4.117      0.000      -0.224      -0.079
    ZN             0.0415      0.017      2.474      0.014       0.009       0.075
    INDUS          0.0186      0.078      0.239      0.811      -0.135       0.172
    CHAS           5.7381      1.158      4.956      0.000       3.461       8.016
    NOX          -17.8560      4.780     -3.735      0.000     -27.258      -8.454
    RM             6.3880      0.452     14.135      0.000       5.499       7.277
    AGE           -0.0523      0.016     -3.342      0.001      -0.083      -0.022
    DIS           -1.4293      0.247     -5.795      0.000      -1.914      -0.944
    RAD            0.2758      0.084      3.264      0.001       0.110       0.442
    TAX           -0.0108      0.005     -2.246      0.025      -0.020      -0.001
    PTRATIO       -0.9095      0.169     -5.397      0.000      -1.241      -0.578
    B              0.0150      0.003      4.324      0.000       0.008       0.022
    ==============================================================================
    Omnibus:                      165.814   Durbin-Watson:                   1.812
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1042.177
    Skew:                           1.876   Prob(JB):                    4.94e-227
    Kurtosis:                      10.522   Cond. No.                     1.46e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.46e+04. This might indicate that there are
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

    r2 score: 0.6329792476946079
    mse: 33.26760340198422
    rmse: 5.767807503894718
    mae: 3.9105840776705896


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.717
    Model:                            OLS   Adj. R-squared:                  0.709
    Method:                 Least Squares   F-statistic:                     86.92
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           9.52e-88
    Time:                        23:13:04   Log-Likelihood:                 46.551
    No. Observations:                 354   AIC:                            -71.10
    Df Residuals:                     343   BIC:                            -28.54
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       2.9129      0.262     11.100      0.000       2.397       3.429
    CRIM           -0.0120      0.002     -7.660      0.000      -0.015      -0.009
    CHAS            0.2013      0.049      4.101      0.000       0.105       0.298
    NOX            -0.7739      0.198     -3.901      0.000      -1.164      -0.384
    RM              0.2296      0.019     12.130      0.000       0.192       0.267
    DIS            -0.0429      0.009     -4.790      0.000      -0.060      -0.025
    RAD             0.0112      0.003      3.217      0.001       0.004       0.018
    TAX            -0.0004      0.000     -2.318      0.021      -0.001   -6.38e-05
    PTRATIO        -0.0411      0.007     -6.117      0.000      -0.054      -0.028
    B               0.0007      0.000      4.846      0.000       0.000       0.001
    pow(AGE, 2) -2.206e-05    5.7e-06     -3.870      0.000   -3.33e-05   -1.08e-05
    ==============================================================================
    Omnibus:                       92.863   Durbin-Watson:                   1.823
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              551.505
    Skew:                           0.945   Prob(JB):                    1.75e-120
    Kurtosis:                       8.815   Cond. No.                     1.66e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.66e+05. This might indicate that there are
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

    r2 score: 0.6961451201986049
    mse: 27.54210373526579
    rmse: 5.2480571391006965
    mae: 3.338727373682676



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
