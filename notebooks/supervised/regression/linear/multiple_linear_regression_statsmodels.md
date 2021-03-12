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
    Dep. Variable:                      y   R-squared (uncentered):                   0.959
    Model:                            OLS   Adj. R-squared (uncentered):              0.957
    Method:                 Least Squares   F-statistic:                              607.5
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):                   8.84e-227
    Time:                        18:22:11   Log-Likelihood:                         -1068.2
    No. Observations:                 354   AIC:                                      2162.
    Df Residuals:                     341   BIC:                                      2213.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1222      0.059     -2.055      0.041      -0.239      -0.005
    ZN             0.0573      0.017      3.327      0.001       0.023       0.091
    INDUS          0.0078      0.077      0.101      0.920      -0.145       0.160
    CHAS           3.1672      1.076      2.943      0.003       1.051       5.284
    NOX           -3.4283      3.981     -0.861      0.390     -11.259       4.402
    RM             5.4338      0.373     14.557      0.000       4.700       6.168
    AGE           -0.0006      0.017     -0.036      0.971      -0.034       0.033
    DIS           -0.8948      0.232     -3.850      0.000      -1.352      -0.438
    RAD            0.2475      0.084      2.939      0.004       0.082       0.413
    TAX           -0.0100      0.005     -2.045      0.042      -0.020      -0.000
    PTRATIO       -0.2998      0.134     -2.231      0.026      -0.564      -0.036
    B              0.0187      0.003      5.786      0.000       0.012       0.025
    LSTAT         -0.5011      0.063     -7.913      0.000      -0.626      -0.377
    ==============================================================================
    Omnibus:                      138.448   Durbin-Watson:                   2.072
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              910.463
    Skew:                           1.486   Prob(JB):                    1.97e-198
    Kurtosis:                      10.273   Cond. No.                     8.32e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.32e+03. This might indicate that there are
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

    r2 score: 0.6897627962216173
    mse: 25.2144405160688
    rmse: 5.021398263040764
    mae: 3.627824423040478


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.742
    Model:                            OLS   Adj. R-squared:                  0.732
    Method:                 Least Squares   F-statistic:                     75.17
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           1.43e-91
    Time:                        18:22:11   Log-Likelihood:                -1050.5
    No. Observations:                 354   AIC:                             2129.
    Df Residuals:                     340   BIC:                             2183.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         37.1413      6.222      5.970      0.000      24.904      49.379
    CRIM          -0.1237      0.057     -2.185      0.030      -0.235      -0.012
    ZN             0.0539      0.016      3.283      0.001       0.022       0.086
    INDUS          0.0603      0.074      0.811      0.418      -0.086       0.206
    CHAS           2.7922      1.027      2.718      0.007       0.772       4.813
    NOX          -19.7525      4.676     -4.224      0.000     -28.950     -10.555
    RM             3.4517      0.487      7.095      0.000       2.495       4.409
    AGE            0.0085      0.016      0.521      0.603      -0.023       0.040
    DIS           -1.4086      0.238     -5.929      0.000      -1.876      -0.941
    RAD            0.3640      0.083      4.407      0.000       0.202       0.526
    TAX           -0.0132      0.005     -2.814      0.005      -0.022      -0.004
    PTRATIO       -0.8822      0.161     -5.481      0.000      -1.199      -0.566
    B              0.0118      0.003      3.571      0.000       0.005       0.018
    LSTAT         -0.6072      0.063     -9.654      0.000      -0.731      -0.483
    ==============================================================================
    Omnibus:                      126.163   Durbin-Watson:                   2.099
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              552.250
    Skew:                           1.479   Prob(JB):                    1.20e-120
    Kurtosis:                       8.356   Cond. No.                     1.54e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.54e+04. This might indicate that there are
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

    r2 score: 0.7229261529519866
    mse: 22.519098128350528
    rmse: 4.745429182734743
    mae: 3.429328344533135


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
      <th>258</th>
      <td>0.66351</td>
      <td>20.0</td>
      <td>3.97</td>
      <td>0.0</td>
      <td>0.647</td>
      <td>7.333</td>
      <td>100.0</td>
      <td>1.8946</td>
      <td>5.0</td>
      <td>264.0</td>
      <td>13.0</td>
      <td>383.29</td>
      <td>7.79</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>400</th>
      <td>25.04610</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.693</td>
      <td>5.987</td>
      <td>100.0</td>
      <td>1.5888</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>26.77</td>
      <td>5.6</td>
    </tr>
    <tr>
      <th>127</th>
      <td>0.25915</td>
      <td>0.0</td>
      <td>21.89</td>
      <td>0.0</td>
      <td>0.624</td>
      <td>5.693</td>
      <td>96.0</td>
      <td>1.7883</td>
      <td>4.0</td>
      <td>437.0</td>
      <td>21.2</td>
      <td>392.11</td>
      <td>17.19</td>
      <td>16.2</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.671
    Model:                            OLS   Adj. R-squared:                  0.660
    Method:                 Least Squares   F-statistic:                     57.99
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           8.16e-75
    Time:                        18:22:12   Log-Likelihood:                -1093.4
    No. Observations:                 354   AIC:                             2213.
    Df Residuals:                     341   BIC:                             2263.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     20.1623      6.726      2.997      0.003       6.932      33.393
    CRIM          -0.2567      0.062     -4.147      0.000      -0.379      -0.135
    ZN             0.0456      0.018      2.465      0.014       0.009       0.082
    INDUS         -0.0308      0.083     -0.370      0.711      -0.194       0.133
    CHAS           3.3995      1.156      2.942      0.003       1.127       5.672
    NOX          -20.0147      5.270     -3.798      0.000     -30.381      -9.649
    RM             5.8196      0.474     12.288      0.000       4.888       6.751
    AGE           -0.0511      0.017     -3.017      0.003      -0.084      -0.018
    DIS           -1.5605      0.267     -5.840      0.000      -2.086      -1.035
    RAD            0.3601      0.093      3.868      0.000       0.177       0.543
    TAX           -0.0120      0.005     -2.266      0.024      -0.022      -0.002
    PTRATIO       -0.9711      0.181     -5.362      0.000      -1.327      -0.615
    B              0.0172      0.004      4.709      0.000       0.010       0.024
    ==============================================================================
    Omnibus:                      207.990   Durbin-Watson:                   2.079
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2062.616
    Skew:                           2.297   Prob(JB):                         0.00
    Kurtosis:                      13.896   Cond. No.                     1.51e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.51e+04. This might indicate that there are
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

    r2 score: 0.6901431107746827
    mse: 25.183530558917074
    rmse: 5.018319495500169
    mae: 3.547336296832994


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.705
    Model:                            OLS   Adj. R-squared:                  0.697
    Method:                 Least Squares   F-statistic:                     82.02
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           1.05e-84
    Time:                        18:22:12   Log-Likelihood:                 27.815
    No. Observations:                 354   AIC:                            -33.63
    Df Residuals:                     343   BIC:                             8.932
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.1554      0.282     11.181      0.000       2.600       3.711
    CRIM           -0.0215      0.003     -8.310      0.000      -0.027      -0.016
    CHAS            0.1135      0.048      2.362      0.019       0.019       0.208
    NOX            -0.8502      0.215     -3.963      0.000      -1.272      -0.428
    RM              0.2056      0.019     10.603      0.000       0.167       0.244
    DIS            -0.0482      0.010     -4.854      0.000      -0.068      -0.029
    RAD             0.0165      0.004      4.378      0.000       0.009       0.024
    TAX            -0.0005      0.000     -2.612      0.009      -0.001      -0.000
    PTRATIO        -0.0444      0.007     -6.313      0.000      -0.058      -0.031
    B               0.0008      0.000      5.487      0.000       0.001       0.001
    pow(AGE, 2) -2.247e-05   5.87e-06     -3.829      0.000    -3.4e-05   -1.09e-05
    ==============================================================================
    Omnibus:                      114.363   Durbin-Watson:                   2.057
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              740.432
    Skew:                           1.184   Prob(JB):                    1.65e-161
    Kurtosis:                       9.678   Cond. No.                     1.79e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.79e+05. This might indicate that there are
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

    r2 score: 0.744415631513719
    mse: 20.772546869130164
    rmse: 4.557690958054327
    mae: 3.06295147898061



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
