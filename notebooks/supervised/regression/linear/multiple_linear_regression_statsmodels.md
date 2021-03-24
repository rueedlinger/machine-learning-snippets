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
    Dep. Variable:                      y   R-squared (uncentered):                   0.961
    Model:                            OLS   Adj. R-squared (uncentered):              0.960
    Method:                 Least Squares   F-statistic:                              647.6
    Date:                Wed, 24 Mar 2021   Prob (F-statistic):                   2.53e-231
    Time:                        16:41:58   Log-Likelihood:                         -1057.7
    No. Observations:                 354   AIC:                                      2141.
    Df Residuals:                     341   BIC:                                      2192.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0877      0.040     -2.181      0.030      -0.167      -0.009
    ZN             0.0498      0.017      2.946      0.003       0.017       0.083
    INDUS         -0.0023      0.077     -0.030      0.976      -0.155       0.150
    CHAS           3.5306      1.119      3.156      0.002       1.330       5.731
    NOX           -0.2665      4.077     -0.065      0.948      -8.285       7.752
    RM             5.2992      0.361     14.662      0.000       4.588       6.010
    AGE           -0.0023      0.016     -0.141      0.888      -0.034       0.029
    DIS           -0.8429      0.221     -3.816      0.000      -1.277      -0.408
    RAD            0.1089      0.084      1.296      0.196      -0.056       0.274
    TAX           -0.0056      0.005     -1.154      0.249      -0.015       0.004
    PTRATIO       -0.3494      0.131     -2.659      0.008      -0.608      -0.091
    B              0.0174      0.003      5.357      0.000       0.011       0.024
    LSTAT         -0.4991      0.059     -8.523      0.000      -0.614      -0.384
    ==============================================================================
    Omnibus:                      133.235   Durbin-Watson:                   1.928
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              813.868
    Skew:                           1.445   Prob(JB):                    1.87e-177
    Kurtosis:                       9.843   Cond. No.                     8.75e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.75e+03. This might indicate that there are
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

    r2 score: 0.7015757990564184
    mse: 27.974716397050255
    rmse: 5.289113006643955
    mae: 3.710111179239169


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.745
    Model:                            OLS   Adj. R-squared:                  0.735
    Method:                 Least Squares   F-statistic:                     76.46
    Date:                Wed, 24 Mar 2021   Prob (F-statistic):           1.70e-92
    Time:                        16:41:59   Log-Likelihood:                -1036.8
    No. Observations:                 354   AIC:                             2102.
    Df Residuals:                     340   BIC:                             2156.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         38.5291      5.910      6.519      0.000      26.904      50.154
    CRIM          -0.1015      0.038     -2.670      0.008      -0.176      -0.027
    ZN             0.0421      0.016      2.631      0.009       0.011       0.074
    INDUS          0.0266      0.073      0.363      0.717      -0.117       0.171
    CHAS           3.5594      1.056      3.370      0.001       1.482       5.637
    NOX          -17.2625      4.649     -3.713      0.000     -26.407      -8.118
    RM             3.2385      0.465      6.962      0.000       2.324       4.153
    AGE            0.0042      0.015      0.280      0.780      -0.025       0.034
    DIS           -1.3895      0.225     -6.182      0.000      -1.832      -0.947
    RAD            0.2620      0.083      3.166      0.002       0.099       0.425
    TAX           -0.0092      0.005     -1.979      0.049      -0.018   -5.62e-05
    PTRATIO       -0.9508      0.155     -6.150      0.000      -1.255      -0.647
    B              0.0109      0.003      3.398      0.001       0.005       0.017
    LSTAT         -0.5994      0.057    -10.443      0.000      -0.712      -0.486
    ==============================================================================
    Omnibus:                      115.223   Durbin-Watson:                   1.910
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              433.465
    Skew:                           1.394   Prob(JB):                     7.49e-95
    Kurtosis:                       7.650   Cond. No.                     1.53e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.53e+04. This might indicate that there are
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

    r2 score: 0.7219683351944068
    mse: 26.063090552788797
    rmse: 5.105202302826872
    mae: 3.673417260913135


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
      <th>163</th>
      <td>1.51902</td>
      <td>0.0</td>
      <td>19.58</td>
      <td>1.0</td>
      <td>0.605</td>
      <td>8.375</td>
      <td>93.9</td>
      <td>2.1620</td>
      <td>5.0</td>
      <td>403.0</td>
      <td>14.7</td>
      <td>388.45</td>
      <td>3.32</td>
      <td>50.0</td>
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
      <th>485</th>
      <td>3.67367</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.583</td>
      <td>6.312</td>
      <td>51.9</td>
      <td>3.9917</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>388.62</td>
      <td>10.58</td>
      <td>21.2</td>
    </tr>
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
    Dep. Variable:                   MEDV   R-squared:                       0.663
    Model:                            OLS   Adj. R-squared:                  0.652
    Method:                 Least Squares   F-statistic:                     56.00
    Date:                Wed, 24 Mar 2021   Prob (F-statistic):           4.11e-73
    Time:                        16:41:59   Log-Likelihood:                -1086.1
    No. Observations:                 354   AIC:                             2198.
    Df Residuals:                     341   BIC:                             2248.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     21.9896      6.534      3.365      0.001       9.137      34.842
    CRIM          -0.1573      0.043     -3.641      0.000      -0.242      -0.072
    ZN             0.0345      0.018      1.880      0.061      -0.002       0.071
    INDUS         -0.0423      0.084     -0.505      0.614      -0.207       0.122
    CHAS           3.7181      1.212      3.068      0.002       1.334       6.102
    NOX          -20.5016      5.323     -3.851      0.000     -30.972     -10.031
    RM             5.6935      0.461     12.361      0.000       4.788       6.599
    AGE           -0.0420      0.017     -2.545      0.011      -0.075      -0.010
    DIS           -1.4614      0.258     -5.669      0.000      -1.969      -0.954
    RAD            0.2050      0.095      2.163      0.031       0.019       0.391
    TAX           -0.0074      0.005     -1.391      0.165      -0.018       0.003
    PTRATIO       -1.0535      0.177     -5.950      0.000      -1.402      -0.705
    B              0.0153      0.004      4.181      0.000       0.008       0.023
    ==============================================================================
    Omnibus:                      196.347   Durbin-Watson:                   1.853
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1834.581
    Skew:                           2.149   Prob(JB):                         0.00
    Kurtosis:                      13.291   Cond. No.                     1.50e+04
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

    r2 score: 0.717718413355332
    mse: 26.46148437534687
    rmse: 5.144072742034941
    mae: 3.557844858934403


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.688
    Model:                            OLS   Adj. R-squared:                  0.678
    Method:                 Least Squares   F-statistic:                     75.49
    Date:                Wed, 24 Mar 2021   Prob (F-statistic):           1.91e-80
    Time:                        16:41:59   Log-Likelihood:                 25.601
    No. Observations:                 354   AIC:                            -29.20
    Df Residuals:                     343   BIC:                             13.36
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.2228      0.281     11.466      0.000       2.670       3.776
    CRIM           -0.0124      0.002     -6.637      0.000      -0.016      -0.009
    CHAS            0.1358      0.052      2.616      0.009       0.034       0.238
    NOX            -1.0096      0.225     -4.480      0.000      -1.453      -0.566
    RM              0.2074      0.019     10.663      0.000       0.169       0.246
    DIS            -0.0455      0.010     -4.549      0.000      -0.065      -0.026
    RAD             0.0076      0.004      1.929      0.055      -0.000       0.015
    TAX            -0.0003      0.000     -1.309      0.191      -0.001       0.000
    PTRATIO        -0.0473      0.007     -6.764      0.000      -0.061      -0.034
    B               0.0008      0.000      4.981      0.000       0.000       0.001
    pow(AGE, 2) -1.728e-05   6.12e-06     -2.826      0.005   -2.93e-05   -5.25e-06
    ==============================================================================
    Omnibus:                       94.126   Durbin-Watson:                   1.812
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              732.614
    Skew:                           0.869   Prob(JB):                    8.22e-160
    Kurtosis:                       9.830   Cond. No.                     1.71e+05
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

    r2 score: 0.7294189475226771
    mse: 25.364659372580558
    rmse: 5.036333921870209
    mae: 3.1985911211140916



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
