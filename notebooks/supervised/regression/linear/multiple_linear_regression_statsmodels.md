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
    Dep. Variable:                      y   R-squared (uncentered):                   0.955
    Model:                            OLS   Adj. R-squared (uncentered):              0.953
    Method:                 Least Squares   F-statistic:                              551.9
    Date:                Mon, 29 Mar 2021   Prob (F-statistic):                   5.32e-220
    Time:                        09:32:26   Log-Likelihood:                         -1086.5
    No. Observations:                 354   AIC:                                      2199.
    Df Residuals:                     341   BIC:                                      2249.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1163      0.043     -2.727      0.007      -0.200      -0.032
    ZN             0.0611      0.019      3.300      0.001       0.025       0.098
    INDUS         -0.0572      0.082     -0.694      0.488      -0.219       0.105
    CHAS           3.5504      1.180      3.008      0.003       1.229       5.872
    NOX           -5.3012      4.411     -1.202      0.230     -13.978       3.376
    RM             5.6995      0.398     14.320      0.000       4.917       6.482
    AGE           -0.0074      0.018     -0.419      0.675      -0.042       0.027
    DIS           -1.2923      0.264     -4.904      0.000      -1.811      -0.774
    RAD            0.1794      0.083      2.168      0.031       0.017       0.342
    TAX           -0.0094      0.005     -1.949      0.052      -0.019    8.78e-05
    PTRATIO       -0.1698      0.146     -1.166      0.244      -0.456       0.117
    B              0.0153      0.003      4.527      0.000       0.009       0.022
    LSTAT         -0.4019      0.064     -6.312      0.000      -0.527      -0.277
    ==============================================================================
    Omnibus:                      142.264   Durbin-Watson:                   2.014
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              844.094
    Skew:                           1.575   Prob(JB):                    5.10e-184
    Kurtosis:                       9.878   Cond. No.                     8.82e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.82e+03. This might indicate that there are
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
    RAD         True
    TAX        False
    PTRATIO    False
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

    r2 score: 0.7614479555846028
    mse: 18.420111542574933
    rmse: 4.291865741443333
    mae: 3.0952379650827133


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.717
    Model:                            OLS   Adj. R-squared:                  0.707
    Method:                 Least Squares   F-statistic:                     66.40
    Date:                Mon, 29 Mar 2021   Prob (F-statistic):           5.75e-85
    Time:                        09:32:26   Log-Likelihood:                -1070.1
    No. Observations:                 354   AIC:                             2168.
    Df Residuals:                     340   BIC:                             2222.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         36.6156      6.364      5.754      0.000      24.098      49.133
    CRIM          -0.1320      0.041     -3.232      0.001      -0.212      -0.052
    ZN             0.0594      0.018      3.353      0.001       0.025       0.094
    INDUS         -0.0299      0.079     -0.379      0.705      -0.185       0.125
    CHAS           3.4762      1.128      3.081      0.002       1.257       5.696
    NOX          -19.7386      4.907     -4.022      0.000     -29.391     -10.086
    RM             3.5431      0.534      6.634      0.000       2.493       4.594
    AGE            0.0013      0.017      0.079      0.937      -0.032       0.035
    DIS           -1.7570      0.265     -6.641      0.000      -2.277      -1.237
    RAD            0.3335      0.084      3.994      0.000       0.169       0.498
    TAX           -0.0132      0.005     -2.831      0.005      -0.022      -0.004
    PTRATIO       -0.7433      0.171     -4.342      0.000      -1.080      -0.407
    B              0.0099      0.003      2.941      0.003       0.003       0.016
    LSTAT         -0.5133      0.064     -8.037      0.000      -0.639      -0.388
    ==============================================================================
    Omnibus:                      124.533   Durbin-Watson:                   1.966
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              476.128
    Skew:                           1.513   Prob(JB):                    4.08e-104
    Kurtosis:                       7.809   Cond. No.                     1.48e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.48e+04. This might indicate that there are
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

    r2 score: 0.7877487367933405
    mse: 16.389261944496454
    rmse: 4.04836534227044
    mae: 2.9669592901922015


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
      <th>391</th>
      <td>5.29305</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.700</td>
      <td>6.051</td>
      <td>82.5</td>
      <td>2.1678</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>378.38</td>
      <td>18.76</td>
      <td>23.2</td>
    </tr>
    <tr>
      <th>290</th>
      <td>0.03502</td>
      <td>80.0</td>
      <td>4.95</td>
      <td>0.0</td>
      <td>0.411</td>
      <td>6.861</td>
      <td>27.9</td>
      <td>5.1167</td>
      <td>4.0</td>
      <td>245.0</td>
      <td>19.2</td>
      <td>396.90</td>
      <td>3.33</td>
      <td>28.5</td>
    </tr>
    <tr>
      <th>291</th>
      <td>0.07886</td>
      <td>80.0</td>
      <td>4.95</td>
      <td>0.0</td>
      <td>0.411</td>
      <td>7.148</td>
      <td>27.7</td>
      <td>5.1167</td>
      <td>4.0</td>
      <td>245.0</td>
      <td>19.2</td>
      <td>396.90</td>
      <td>3.56</td>
      <td>37.3</td>
    </tr>
    <tr>
      <th>370</th>
      <td>6.53876</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>1.0</td>
      <td>0.631</td>
      <td>7.016</td>
      <td>97.5</td>
      <td>1.2024</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>392.05</td>
      <td>2.96</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>221</th>
      <td>0.40771</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>1.0</td>
      <td>0.507</td>
      <td>6.164</td>
      <td>91.3</td>
      <td>3.0480</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>395.24</td>
      <td>21.46</td>
      <td>21.7</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.664
    Model:                            OLS   Adj. R-squared:                  0.652
    Method:                 Least Squares   F-statistic:                     56.09
    Date:                Mon, 29 Mar 2021   Prob (F-statistic):           3.41e-73
    Time:                        09:32:26   Log-Likelihood:                -1100.9
    No. Observations:                 354   AIC:                             2228.
    Df Residuals:                     341   BIC:                             2278.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     21.1075      6.605      3.196      0.002       8.115      34.100
    CRIM          -0.1828      0.044     -4.156      0.000      -0.269      -0.096
    ZN             0.0507      0.019      2.632      0.009       0.013       0.089
    INDUS         -0.0739      0.086     -0.862      0.389      -0.243       0.095
    CHAS           4.2848      1.224      3.500      0.001       1.877       6.693
    NOX          -23.4888      5.321     -4.414      0.000     -33.955     -13.023
    RM             5.8844      0.488     12.069      0.000       4.925       6.843
    AGE           -0.0367      0.018     -2.072      0.039      -0.071      -0.002
    DIS           -1.7708      0.288     -6.145      0.000      -2.338      -1.204
    RAD            0.3333      0.091      3.664      0.000       0.154       0.512
    TAX           -0.0135      0.005     -2.652      0.008      -0.023      -0.003
    PTRATIO       -0.8257      0.186     -4.436      0.000      -1.192      -0.460
    B              0.0138      0.004      3.817      0.000       0.007       0.021
    ==============================================================================
    Omnibus:                      178.162   Durbin-Watson:                   1.985
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1410.118
    Skew:                           1.955   Prob(JB):                    6.26e-307
    Kurtosis:                      11.962   Cond. No.                     1.45e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.45e+04. This might indicate that there are
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

    r2 score: 0.72246867096123
    mse: 21.429948546369467
    rmse: 4.6292492421957006
    mae: 3.19256526339842


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.684
    Model:                            OLS   Adj. R-squared:                  0.674
    Method:                 Least Squares   F-statistic:                     74.11
    Date:                Mon, 29 Mar 2021   Prob (F-statistic):           1.63e-79
    Time:                        09:32:26   Log-Likelihood:                 7.1501
    No. Observations:                 354   AIC:                             7.700
    Df Residuals:                     343   BIC:                             50.26
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.1747      0.288     11.007      0.000       2.607       3.742
    CRIM           -0.0139      0.002     -7.296      0.000      -0.018      -0.010
    CHAS            0.1597      0.053      3.011      0.003       0.055       0.264
    NOX            -1.0723      0.223     -4.801      0.000      -1.512      -0.633
    RM              0.2211      0.021     10.653      0.000       0.180       0.262
    DIS            -0.0477      0.010     -4.566      0.000      -0.068      -0.027
    RAD             0.0148      0.004      3.879      0.000       0.007       0.022
    TAX            -0.0006      0.000     -3.139      0.002      -0.001      -0.000
    PTRATIO        -0.0409      0.008     -5.418      0.000      -0.056      -0.026
    B               0.0007      0.000      4.202      0.000       0.000       0.001
    pow(AGE, 2) -1.595e-05   6.51e-06     -2.448      0.015   -2.88e-05   -3.14e-06
    ==============================================================================
    Omnibus:                       71.860   Durbin-Watson:                   2.033
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              470.404
    Skew:                           0.644   Prob(JB):                    7.13e-103
    Kurtosis:                       8.498   Cond. No.                     1.68e+05
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

    r2 score: 0.7955942063658261
    mse: 15.783463637534908
    rmse: 3.9728407515951236
    mae: 2.68204755174146



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_1.png)
    
