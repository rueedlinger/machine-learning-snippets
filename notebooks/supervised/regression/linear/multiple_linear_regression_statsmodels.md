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
    Dep. Variable:                      y   R-squared (uncentered):                   0.958
    Model:                            OLS   Adj. R-squared (uncentered):              0.956
    Method:                 Least Squares   F-statistic:                              598.1
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):                   1.11e-225
    Time:                        18:40:44   Log-Likelihood:                         -1074.3
    No. Observations:                 354   AIC:                                      2175.
    Df Residuals:                     341   BIC:                                      2225.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0214      0.060     -0.355      0.723      -0.140       0.097
    ZN             0.0421      0.017      2.409      0.017       0.008       0.077
    INDUS         -0.0106      0.086     -0.124      0.902      -0.179       0.158
    CHAS           3.9383      1.040      3.786      0.000       1.892       5.985
    NOX           -1.1668      3.988     -0.293      0.770      -9.010       6.677
    RM             5.6525      0.376     15.017      0.000       4.912       6.393
    AGE           -0.0068      0.017     -0.407      0.685      -0.040       0.026
    DIS           -0.8640      0.235     -3.672      0.000      -1.327      -0.401
    RAD            0.0876      0.091      0.968      0.334      -0.090       0.266
    TAX           -0.0076      0.005     -1.428      0.154      -0.018       0.003
    PTRATIO       -0.3137      0.131     -2.386      0.018      -0.572      -0.055
    B              0.0134      0.003      4.120      0.000       0.007       0.020
    LSTAT         -0.4536      0.060     -7.589      0.000      -0.571      -0.336
    ==============================================================================
    Omnibus:                      148.248   Durbin-Watson:                   2.139
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              743.074
    Skew:                           1.725   Prob(JB):                    4.40e-162
    Kurtosis:                       9.203   Cond. No.                     8.18e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.18e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_9_0.png)
    



```python
result.pvalues < 0.05
```




    CRIM       False
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

    r2 score: 0.7073538674369784
    mse: 22.85619342495314
    rmse: 4.780815142311313
    mae: 3.1849757677016854


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.734
    Model:                            OLS   Adj. R-squared:                  0.724
    Method:                 Least Squares   F-statistic:                     72.18
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           2.19e-89
    Time:                        18:40:44   Log-Likelihood:                -1058.4
    No. Observations:                 354   AIC:                             2145.
    Df Residuals:                     340   BIC:                             2199.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         35.8882      6.342      5.659      0.000      23.414      48.362
    CRIM          -0.0694      0.058     -1.192      0.234      -0.184       0.045
    ZN             0.0385      0.017      2.299      0.022       0.006       0.071
    INDUS          0.0126      0.082      0.154      0.878      -0.149       0.174
    CHAS           3.5603      0.998      3.567      0.000       1.597       5.524
    NOX          -16.0424      4.635     -3.461      0.001     -25.160      -6.925
    RM             3.6866      0.501      7.365      0.000       2.702       4.671
    AGE           -0.0012      0.016     -0.073      0.942      -0.033       0.030
    DIS           -1.3394      0.240     -5.571      0.000      -1.812      -0.866
    RAD            0.2532      0.091      2.769      0.006       0.073       0.433
    TAX           -0.0109      0.005     -2.139      0.033      -0.021      -0.001
    PTRATIO       -0.9246      0.166     -5.576      0.000      -1.251      -0.598
    B              0.0085      0.003      2.630      0.009       0.002       0.015
    LSTAT         -0.5366      0.059     -9.084      0.000      -0.653      -0.420
    ==============================================================================
    Omnibus:                      122.636   Durbin-Watson:                   2.077
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              421.439
    Skew:                           1.533   Prob(JB):                     3.06e-92
    Kurtosis:                       7.378   Cond. No.                     1.51e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.51e+04. This might indicate that there are
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

    r2 score: 0.7493141579732621
    mse: 19.579018673778197
    rmse: 4.424818490489548
    mae: 3.0792517403655406


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
      <th>327</th>
      <td>0.24103</td>
      <td>0.0</td>
      <td>7.38</td>
      <td>0.0</td>
      <td>0.493</td>
      <td>6.083</td>
      <td>43.7</td>
      <td>5.4159</td>
      <td>5.0</td>
      <td>287.0</td>
      <td>19.6</td>
      <td>396.90</td>
      <td>12.79</td>
      <td>22.2</td>
    </tr>
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
      <th>23</th>
      <td>0.98843</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.813</td>
      <td>100.0</td>
      <td>4.0952</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>394.54</td>
      <td>19.88</td>
      <td>14.5</td>
    </tr>
    <tr>
      <th>249</th>
      <td>0.19073</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.431</td>
      <td>6.718</td>
      <td>17.5</td>
      <td>7.8265</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>393.74</td>
      <td>6.56</td>
      <td>26.2</td>
    </tr>
    <tr>
      <th>276</th>
      <td>0.10469</td>
      <td>40.0</td>
      <td>6.41</td>
      <td>1.0</td>
      <td>0.447</td>
      <td>7.267</td>
      <td>49.0</td>
      <td>4.7872</td>
      <td>4.0</td>
      <td>254.0</td>
      <td>17.6</td>
      <td>389.25</td>
      <td>6.05</td>
      <td>33.2</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.669
    Model:                            OLS   Adj. R-squared:                  0.658
    Method:                 Least Squares   F-statistic:                     57.56
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           1.88e-74
    Time:                        18:40:44   Log-Likelihood:                -1096.9
    No. Observations:                 354   AIC:                             2220.
    Df Residuals:                     341   BIC:                             2270.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     21.5872      6.838      3.157      0.002       8.137      35.037
    CRIM          -0.1782      0.063     -2.809      0.005      -0.303      -0.053
    ZN             0.0345      0.019      1.852      0.065      -0.002       0.071
    INDUS         -0.0689      0.091     -0.760      0.448      -0.247       0.110
    CHAS           4.3828      1.107      3.961      0.000       2.206       6.559
    NOX          -19.1878      5.145     -3.729      0.000     -29.308      -9.068
    RM             5.9742      0.482     12.407      0.000       5.027       6.921
    AGE           -0.0505      0.017     -3.010      0.003      -0.083      -0.017
    DIS           -1.4789      0.267     -5.537      0.000      -2.004      -0.954
    RAD            0.2837      0.102      2.788      0.006       0.084       0.484
    TAX           -0.0102      0.006     -1.788      0.075      -0.021       0.001
    PTRATIO       -1.0137      0.184     -5.502      0.000      -1.376      -0.651
    B              0.0114      0.004      3.192      0.002       0.004       0.018
    ==============================================================================
    Omnibus:                      188.402   Durbin-Watson:                   1.979
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1505.308
    Skew:                           2.099   Prob(JB):                         0.00
    Kurtosis:                      12.189   Cond. No.                     1.49e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.49e+04. This might indicate that there are
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

    r2 score: 0.7136337400099237
    mse: 22.36572399364984
    rmse: 4.729241376124699
    mae: 3.1926139566722282


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.688
    Model:                            OLS   Adj. R-squared:                  0.679
    Method:                 Least Squares   F-statistic:                     75.57
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           1.69e-80
    Time:                        18:40:45   Log-Likelihood:                 15.516
    No. Observations:                 354   AIC:                            -9.032
    Df Residuals:                     343   BIC:                             33.53
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.3459      0.295     11.351      0.000       2.766       3.926
    CRIM           -0.0167      0.003     -6.142      0.000      -0.022      -0.011
    CHAS            0.1644      0.047      3.491      0.001       0.072       0.257
    NOX            -0.9325      0.215     -4.331      0.000      -1.356      -0.509
    RM              0.2060      0.020     10.215      0.000       0.166       0.246
    DIS            -0.0448      0.010     -4.452      0.000      -0.065      -0.025
    RAD             0.0143      0.004      3.359      0.001       0.006       0.023
    TAX            -0.0005      0.000     -2.381      0.018      -0.001      -9e-05
    PTRATIO        -0.0467      0.007     -6.247      0.000      -0.061      -0.032
    B               0.0005      0.000      3.369      0.001       0.000       0.001
    pow(AGE, 2) -2.121e-05   6.04e-06     -3.510      0.001   -3.31e-05   -9.33e-06
    ==============================================================================
    Omnibus:                       87.340   Durbin-Watson:                   1.952
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              542.104
    Skew:                           0.861   Prob(JB):                    1.92e-118
    Kurtosis:                       8.813   Cond. No.                     1.77e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.77e+05. This might indicate that there are
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

    r2 score: 0.7237229990480161
    mse: 21.577734574246062
    rmse: 4.645184019416891
    mae: 2.914544624028764



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
