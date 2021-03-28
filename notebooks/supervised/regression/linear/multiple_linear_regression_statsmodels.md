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
    Dep. Variable:                      y   R-squared (uncentered):                   0.966
    Model:                            OLS   Adj. R-squared (uncentered):              0.965
    Method:                 Least Squares   F-statistic:                              742.9
    Date:                Sun, 28 Mar 2021   Prob (F-statistic):                   4.21e-241
    Time:                        22:06:17   Log-Likelihood:                         -1036.8
    No. Observations:                 354   AIC:                                      2100.
    Df Residuals:                     341   BIC:                                      2150.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0995      0.035     -2.867      0.004      -0.168      -0.031
    ZN             0.0445      0.016      2.793      0.006       0.013       0.076
    INDUS          0.0287      0.070      0.412      0.681      -0.109       0.166
    CHAS           2.5976      1.045      2.486      0.013       0.542       4.653
    NOX           -4.8928      3.717     -1.316      0.189     -12.204       2.419
    RM             6.6648      0.356     18.720      0.000       5.965       7.365
    AGE           -0.0291      0.015     -1.923      0.055      -0.059       0.001
    DIS           -1.0878      0.213     -5.114      0.000      -1.506      -0.669
    RAD            0.1385      0.075      1.844      0.066      -0.009       0.286
    TAX           -0.0087      0.004     -1.964      0.050      -0.017    1.33e-05
    PTRATIO       -0.5373      0.124     -4.328      0.000      -0.782      -0.293
    B              0.0128      0.003      4.074      0.000       0.007       0.019
    LSTAT         -0.2788      0.058     -4.833      0.000      -0.392      -0.165
    ==============================================================================
    Omnibus:                      178.084   Durbin-Watson:                   1.970
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1320.174
    Skew:                           1.981   Prob(JB):                    2.13e-287
    Kurtosis:                      11.591   Cond. No.                     8.46e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.46e+03. This might indicate that there are
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

    r2 score: 0.6003548192049106
    mse: 34.61611699571974
    rmse: 5.883546294176646
    mae: 3.756590178075367


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.767
    Model:                            OLS   Adj. R-squared:                  0.758
    Method:                 Least Squares   F-statistic:                     86.27
    Date:                Sun, 28 Mar 2021   Prob (F-statistic):           3.66e-99
    Time:                        22:06:17   Log-Likelihood:                -1027.0
    No. Observations:                 354   AIC:                             2082.
    Df Residuals:                     340   BIC:                             2136.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         25.9145      5.901      4.392      0.000      14.307      37.521
    CRIM          -0.1059      0.034     -3.129      0.002      -0.172      -0.039
    ZN             0.0439      0.016      2.827      0.005       0.013       0.074
    INDUS          0.0310      0.068      0.456      0.649      -0.103       0.165
    CHAS           2.3638      1.020      2.319      0.021       0.358       4.369
    NOX          -14.7380      4.259     -3.460      0.001     -23.115      -6.361
    RM             5.0850      0.500     10.176      0.000       4.102       6.068
    AGE           -0.0230      0.015     -1.556      0.121      -0.052       0.006
    DIS           -1.4498      0.223     -6.501      0.000      -1.888      -1.011
    RAD            0.2311      0.076      3.035      0.003       0.081       0.381
    TAX           -0.0105      0.004     -2.438      0.015      -0.019      -0.002
    PTRATIO       -0.9178      0.149     -6.169      0.000      -1.210      -0.625
    B              0.0087      0.003      2.714      0.007       0.002       0.015
    LSTAT         -0.3689      0.060     -6.165      0.000      -0.487      -0.251
    ==============================================================================
    Omnibus:                      161.414   Durbin-Watson:                   2.037
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              955.332
    Skew:                           1.839   Prob(JB):                    3.57e-208
    Kurtosis:                      10.158   Cond. No.                     1.52e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.52e+04. This might indicate that there are
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

    r2 score: 0.6535649976590743
    mse: 30.007204262009406
    rmse: 5.47788319170913
    mae: 3.6869996286748674


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
      <th>373</th>
      <td>11.10810</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.668</td>
      <td>4.906</td>
      <td>100.0</td>
      <td>1.1742</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>34.77</td>
      <td>13.8</td>
    </tr>
    <tr>
      <th>274</th>
      <td>0.05644</td>
      <td>40.0</td>
      <td>6.41</td>
      <td>1.0</td>
      <td>0.447</td>
      <td>6.758</td>
      <td>32.9</td>
      <td>4.0776</td>
      <td>4.0</td>
      <td>254.0</td>
      <td>17.6</td>
      <td>396.90</td>
      <td>3.53</td>
      <td>32.4</td>
    </tr>
    <tr>
      <th>387</th>
      <td>22.59710</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.700</td>
      <td>5.000</td>
      <td>89.5</td>
      <td>1.5184</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>31.99</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>458</th>
      <td>7.75223</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.301</td>
      <td>83.7</td>
      <td>2.7831</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>272.21</td>
      <td>16.23</td>
      <td>14.9</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.741
    Model:                            OLS   Adj. R-squared:                  0.732
    Method:                 Least Squares   F-statistic:                     81.45
    Date:                Sun, 28 Mar 2021   Prob (F-statistic):           2.19e-92
    Time:                        22:06:17   Log-Likelihood:                -1045.8
    No. Observations:                 354   AIC:                             2118.
    Df Residuals:                     341   BIC:                             2168.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     13.4511      5.837      2.304      0.022       1.970      24.932
    CRIM          -0.1378      0.035     -3.914      0.000      -0.207      -0.069
    ZN             0.0352      0.016      2.159      0.032       0.003       0.067
    INDUS          0.0224      0.072      0.313      0.754      -0.118       0.163
    CHAS           2.5666      1.073      2.392      0.017       0.456       4.677
    NOX          -17.1829      4.465     -3.849      0.000     -25.965      -8.401
    RM             6.9345      0.421     16.480      0.000       6.107       7.762
    AGE           -0.0532      0.015     -3.621      0.000      -0.082      -0.024
    DIS           -1.4068      0.235     -5.995      0.000      -1.868      -0.945
    RAD            0.1955      0.080      2.444      0.015       0.038       0.353
    TAX           -0.0094      0.005     -2.067      0.039      -0.018      -0.000
    PTRATIO       -0.9937      0.156     -6.366      0.000      -1.301      -0.687
    B              0.0114      0.003      3.420      0.001       0.005       0.018
    ==============================================================================
    Omnibus:                      193.130   Durbin-Watson:                   1.954
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1764.139
    Skew:                           2.112   Prob(JB):                         0.00
    Kurtosis:                      13.088   Cond. No.                     1.47e+04
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

    r2 score: 0.5392209055855666
    mse: 39.911360896930425
    rmse: 6.317543897507197
    mae: 3.93466494754723


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.756
    Model:                            OLS   Adj. R-squared:                  0.749
    Method:                 Least Squares   F-statistic:                     106.5
    Date:                Sun, 28 Mar 2021   Prob (F-statistic):           8.65e-99
    Time:                        22:06:17   Log-Likelihood:                 68.323
    No. Observations:                 354   AIC:                            -114.6
    Df Residuals:                     343   BIC:                            -72.08
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       2.8466      0.250     11.375      0.000       2.354       3.339
    CRIM           -0.0120      0.002     -7.963      0.000      -0.015      -0.009
    CHAS            0.1151      0.046      2.515      0.012       0.025       0.205
    NOX            -0.8012      0.187     -4.279      0.000      -1.169      -0.433
    RM              0.2544      0.017     14.541      0.000       0.220       0.289
    DIS            -0.0449      0.009     -5.190      0.000      -0.062      -0.028
    RAD             0.0060      0.003      1.814      0.070      -0.001       0.012
    TAX            -0.0002      0.000     -1.451      0.148      -0.001    8.86e-05
    PTRATIO        -0.0424      0.006     -6.754      0.000      -0.055      -0.030
    B               0.0005      0.000      3.782      0.000       0.000       0.001
    pow(AGE, 2) -2.189e-05    5.3e-06     -4.132      0.000   -3.23e-05   -1.15e-05
    ==============================================================================
    Omnibus:                      107.397   Durbin-Watson:                   2.062
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              876.952
    Skew:                           1.018   Prob(JB):                    3.73e-191
    Kurtosis:                      10.437   Cond. No.                     1.71e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.71e+05. This might indicate that there are
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

    r2 score: 0.6023494787547354
    mse: 34.443345318088106
    rmse: 5.868845313866102
    mae: 3.433559150305023



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_1.png)
    
