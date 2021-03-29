>**Note**: This is a generated markdown export from the Jupyter notebook file [multiple_linear_regression_statsmodels.ipynb](multiple_linear_regression_statsmodels.ipynb).
>You can also view the notebook with the [nbviewer](https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master/notebooks/supervised/regression/linear/multiple_linear_regression_statsmodels.ipynb) from Jupyter. 

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
    Method:                 Least Squares   F-statistic:                              669.2
    Date:                Sun, 28 Mar 2021   Prob (F-statistic):                   1.19e-233
    Time:                        22:24:19   Log-Likelihood:                         -1047.6
    No. Observations:                 354   AIC:                                      2121.
    Df Residuals:                     341   BIC:                                      2171.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0862      0.035     -2.445      0.015      -0.155      -0.017
    ZN             0.0613      0.017      3.545      0.000       0.027       0.095
    INDUS         -0.0293      0.076     -0.387      0.699      -0.178       0.120
    CHAS           3.5441      1.053      3.367      0.001       1.473       5.615
    NOX           -0.6575      3.695     -0.178      0.859      -7.925       6.610
    RM             5.7768      0.360     16.057      0.000       5.069       6.484
    AGE           -0.0166      0.016     -1.049      0.295      -0.048       0.015
    DIS           -1.1092      0.233     -4.762      0.000      -1.567      -0.651
    RAD            0.1327      0.078      1.705      0.089      -0.020       0.286
    TAX           -0.0083      0.005     -1.805      0.072      -0.017       0.001
    PTRATIO       -0.2609      0.132     -1.979      0.049      -0.520      -0.002
    B              0.0110      0.003      3.514      0.001       0.005       0.017
    LSTAT         -0.4367      0.060     -7.225      0.000      -0.556      -0.318
    ==============================================================================
    Omnibus:                      102.521   Durbin-Watson:                   2.059
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              494.759
    Skew:                           1.137   Prob(JB):                    3.67e-108
    Kurtosis:                       8.327   Cond. No.                     8.19e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.19e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")
```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_9_1.png)
    



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

    r2 score: 0.6977994320804599
    mse: 30.587581941864574
    rmse: 5.530604120877264
    mae: 3.646929903684103


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.746
    Model:                            OLS   Adj. R-squared:                  0.736
    Method:                 Least Squares   F-statistic:                     76.73
    Date:                Sun, 28 Mar 2021   Prob (F-statistic):           1.10e-92
    Time:                        22:24:19   Log-Likelihood:                -1029.0
    No. Observations:                 354   AIC:                             2086.
    Df Residuals:                     340   BIC:                             2140.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         35.2317      5.744      6.134      0.000      23.933      46.530
    CRIM          -0.1025      0.034     -3.052      0.002      -0.169      -0.036
    ZN             0.0561      0.016      3.409      0.001       0.024       0.089
    INDUS          0.0246      0.072      0.340      0.734      -0.118       0.167
    CHAS           3.1869      1.002      3.180      0.002       1.216       5.158
    NOX          -15.4848      4.263     -3.632      0.000     -23.870      -7.100
    RM             3.7819      0.472      8.015      0.000       2.854       4.710
    AGE           -0.0083      0.015     -0.552      0.581      -0.038       0.021
    DIS           -1.5338      0.232     -6.613      0.000      -1.990      -1.078
    RAD            0.2629      0.077      3.418      0.001       0.112       0.414
    TAX           -0.0113      0.004     -2.582      0.010      -0.020      -0.003
    PTRATIO       -0.8392      0.157     -5.352      0.000      -1.148      -0.531
    B              0.0057      0.003      1.845      0.066      -0.000       0.012
    LSTAT         -0.5412      0.060     -9.034      0.000      -0.659      -0.423
    ==============================================================================
    Omnibus:                       91.612   Durbin-Watson:                   1.974
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              280.794
    Skew:                           1.162   Prob(JB):                     1.06e-61
    Kurtosis:                       6.693   Cond. No.                     1.50e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.5e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")

```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_14_1.png)
    



```python
predicted = result.predict(sm.add_constant(X_test))

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.7237902848867163
    mse: 27.956887547663534
    rmse: 5.287427308972061
    mae: 3.548416560743556


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
      <th>154</th>
      <td>1.41385</td>
      <td>0.0</td>
      <td>19.58</td>
      <td>1.0</td>
      <td>0.871</td>
      <td>6.129</td>
      <td>96.0</td>
      <td>1.7494</td>
      <td>5.0</td>
      <td>403.0</td>
      <td>14.7</td>
      <td>321.02</td>
      <td>15.12</td>
      <td>17.0</td>
    </tr>
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
      <th>92</th>
      <td>0.04203</td>
      <td>28.0</td>
      <td>15.04</td>
      <td>0.0</td>
      <td>0.464</td>
      <td>6.442</td>
      <td>53.6</td>
      <td>3.6659</td>
      <td>4.0</td>
      <td>270.0</td>
      <td>18.2</td>
      <td>395.01</td>
      <td>8.16</td>
      <td>22.9</td>
    </tr>
    <tr>
      <th>481</th>
      <td>5.70818</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.532</td>
      <td>6.750</td>
      <td>74.9</td>
      <td>3.3317</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>393.07</td>
      <td>7.74</td>
      <td>23.7</td>
    </tr>
    <tr>
      <th>135</th>
      <td>0.55778</td>
      <td>0.0</td>
      <td>21.89</td>
      <td>0.0</td>
      <td>0.624</td>
      <td>6.335</td>
      <td>98.2</td>
      <td>2.1107</td>
      <td>4.0</td>
      <td>437.0</td>
      <td>21.2</td>
      <td>394.67</td>
      <td>16.96</td>
      <td>18.1</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.685
    Model:                            OLS   Adj. R-squared:                  0.674
    Method:                 Least Squares   F-statistic:                     61.73
    Date:                Sun, 28 Mar 2021   Prob (F-statistic):           6.59e-78
    Time:                        22:24:19   Log-Likelihood:                -1067.1
    No. Observations:                 354   AIC:                             2160.
    Df Residuals:                     341   BIC:                             2210.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     20.4713      6.123      3.343      0.001       8.427      32.515
    CRIM          -0.1479      0.037     -4.006      0.000      -0.221      -0.075
    ZN             0.0486      0.018      2.657      0.008       0.013       0.085
    INDUS         -0.0353      0.080     -0.440      0.660      -0.193       0.122
    CHAS           4.2807      1.106      3.870      0.000       2.105       6.456
    NOX          -17.6745      4.732     -3.735      0.000     -26.983      -8.366
    RM             5.8886      0.456     12.909      0.000       4.991       6.786
    AGE           -0.0558      0.016     -3.536      0.000      -0.087      -0.025
    DIS           -1.6102      0.258     -6.248      0.000      -2.117      -1.103
    RAD            0.2630      0.086      3.076      0.002       0.095       0.431
    TAX           -0.0113      0.005     -2.312      0.021      -0.021      -0.002
    PTRATIO       -0.9178      0.174     -5.272      0.000      -1.260      -0.575
    B              0.0109      0.003      3.231      0.001       0.004       0.018
    ==============================================================================
    Omnibus:                      131.226   Durbin-Watson:                   1.961
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              651.261
    Skew:                           1.497   Prob(JB):                    3.81e-142
    Kurtosis:                       8.932   Cond. No.                     1.47e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.47e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")
```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_20_1.png)
    



```python
predicted = result.predict(X_test)

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.6731375617829016
    mse: 33.08376182583834
    rmse: 5.751848557276029
    mae: 3.596084442108686


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.691
    Model:                            OLS   Adj. R-squared:                  0.682
    Method:                 Least Squares   F-statistic:                     76.56
    Date:                Sun, 28 Mar 2021   Prob (F-statistic):           3.67e-81
    Time:                        22:24:19   Log-Likelihood:                 36.985
    No. Observations:                 354   AIC:                            -51.97
    Df Residuals:                     343   BIC:                            -9.408
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.1737      0.269     11.820      0.000       2.646       3.702
    CRIM           -0.0117      0.002     -7.172      0.000      -0.015      -0.008
    CHAS            0.1501      0.048      3.099      0.002       0.055       0.245
    NOX            -0.8230      0.199     -4.127      0.000      -1.215      -0.431
    RM              0.2177      0.020     11.089      0.000       0.179       0.256
    DIS            -0.0483      0.010     -5.002      0.000      -0.067      -0.029
    RAD             0.0110      0.004      3.057      0.002       0.004       0.018
    TAX            -0.0005      0.000     -2.519      0.012      -0.001      -0.000
    PTRATIO        -0.0439      0.007     -6.409      0.000      -0.057      -0.030
    B               0.0005      0.000      3.614      0.000       0.000       0.001
    pow(AGE, 2) -2.261e-05   5.82e-06     -3.884      0.000   -3.41e-05   -1.12e-05
    ==============================================================================
    Omnibus:                       63.154   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              294.744
    Skew:                           0.649   Prob(JB):                     9.94e-65
    Kurtosis:                       7.278   Cond. No.                     1.68e+05
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

    r2 score: 0.7141284509049501
    mse: 28.93482131086098
    rmse: 5.379109713592109
    mae: 3.110173431234217



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_1.png)
    
