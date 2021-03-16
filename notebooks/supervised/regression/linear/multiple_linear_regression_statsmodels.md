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
    Dep. Variable:                      y   R-squared (uncentered):                   0.952
    Model:                            OLS   Adj. R-squared (uncentered):              0.950
    Method:                 Least Squares   F-statistic:                              517.9
    Date:                Tue, 16 Mar 2021   Prob (F-statistic):                   1.66e-215
    Time:                        21:38:49   Log-Likelihood:                         -1098.5
    No. Observations:                 354   AIC:                                      2223.
    Df Residuals:                     341   BIC:                                      2273.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1031      0.046     -2.218      0.027      -0.195      -0.012
    ZN             0.0582      0.019      3.014      0.003       0.020       0.096
    INDUS         -0.0102      0.085     -0.120      0.905      -0.178       0.158
    CHAS           3.5111      1.161      3.025      0.003       1.228       5.794
    NOX           -0.8877      4.175     -0.213      0.832      -9.101       7.325
    RM             5.5065      0.413     13.341      0.000       4.695       6.318
    AGE           -0.0019      0.019     -0.101      0.919      -0.038       0.035
    DIS           -0.9632      0.256     -3.756      0.000      -1.468      -0.459
    RAD            0.1422      0.087      1.634      0.103      -0.029       0.313
    TAX           -0.0075      0.005     -1.455      0.147      -0.018       0.003
    PTRATIO       -0.3714      0.146     -2.548      0.011      -0.658      -0.085
    B              0.0168      0.004      4.776      0.000       0.010       0.024
    LSTAT         -0.4401      0.065     -6.808      0.000      -0.567      -0.313
    ==============================================================================
    Omnibus:                      137.150   Durbin-Watson:                   1.830
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              674.723
    Skew:                           1.580   Prob(JB):                    3.06e-147
    Kurtosis:                       8.980   Cond. No.                     8.03e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.03e+03. This might indicate that there are
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

    r2 score: 0.8156787542786821
    mse: 13.5037574004199
    rmse: 3.67474589603416
    mae: 2.7207018698385634


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.707
    Model:                            OLS   Adj. R-squared:                  0.696
    Method:                 Least Squares   F-statistic:                     63.03
    Date:                Tue, 16 Mar 2021   Prob (F-statistic):           2.90e-82
    Time:                        21:38:49   Log-Likelihood:                -1080.0
    No. Observations:                 354   AIC:                             2188.
    Df Residuals:                     340   BIC:                             2242.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         41.0711      6.702      6.129      0.000      27.889      54.253
    CRIM          -0.1062      0.044     -2.405      0.017      -0.193      -0.019
    ZN             0.0507      0.018      2.756      0.006       0.015       0.087
    INDUS          0.0402      0.082      0.493      0.622      -0.120       0.201
    CHAS           3.0265      1.106      2.737      0.007       0.851       5.202
    NOX          -18.8102      4.929     -3.816      0.000     -28.506      -9.114
    RM             3.3253      0.530      6.278      0.000       2.283       4.367
    AGE            0.0068      0.018      0.384      0.701      -0.028       0.042
    DIS           -1.5134      0.260     -5.827      0.000      -2.024      -1.003
    RAD            0.2884      0.086      3.350      0.001       0.119       0.458
    TAX           -0.0110      0.005     -2.230      0.026      -0.021      -0.001
    PTRATIO       -1.0379      0.176     -5.893      0.000      -1.384      -0.691
    B              0.0097      0.004      2.740      0.006       0.003       0.017
    LSTAT         -0.5550      0.064     -8.641      0.000      -0.681      -0.429
    ==============================================================================
    Omnibus:                      118.978   Durbin-Watson:                   1.852
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              390.738
    Skew:                           1.503   Prob(JB):                     1.42e-85
    Kurtosis:                       7.178   Cond. No.                     1.52e+04
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

    r2 score: 0.8298769101109175
    mse: 12.46357100659667
    rmse: 3.5303783092746124
    mae: 2.6531858265523662


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
      <th>81</th>
      <td>0.04462</td>
      <td>25.0</td>
      <td>4.86</td>
      <td>0.0</td>
      <td>0.426</td>
      <td>6.619</td>
      <td>70.4</td>
      <td>5.4007</td>
      <td>4.0</td>
      <td>281.0</td>
      <td>19.0</td>
      <td>395.63</td>
      <td>7.22</td>
      <td>23.9</td>
    </tr>
    <tr>
      <th>285</th>
      <td>0.01096</td>
      <td>55.0</td>
      <td>2.25</td>
      <td>0.0</td>
      <td>0.389</td>
      <td>6.453</td>
      <td>31.9</td>
      <td>7.3073</td>
      <td>1.0</td>
      <td>300.0</td>
      <td>15.3</td>
      <td>394.72</td>
      <td>8.23</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0.01360</td>
      <td>75.0</td>
      <td>4.00</td>
      <td>0.0</td>
      <td>0.410</td>
      <td>5.888</td>
      <td>47.6</td>
      <td>7.3197</td>
      <td>3.0</td>
      <td>469.0</td>
      <td>21.1</td>
      <td>396.90</td>
      <td>14.80</td>
      <td>18.9</td>
    </tr>
    <tr>
      <th>380</th>
      <td>88.97620</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.671</td>
      <td>6.968</td>
      <td>91.9</td>
      <td>1.4165</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>17.21</td>
      <td>10.4</td>
    </tr>
    <tr>
      <th>245</th>
      <td>0.19133</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.431</td>
      <td>5.605</td>
      <td>70.2</td>
      <td>7.9549</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>389.13</td>
      <td>18.46</td>
      <td>18.5</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.642
    Model:                            OLS   Adj. R-squared:                  0.630
    Method:                 Least Squares   F-statistic:                     51.04
    Date:                Tue, 16 Mar 2021   Prob (F-statistic):           1.07e-68
    Time:                        21:38:49   Log-Likelihood:                -1115.1
    No. Observations:                 354   AIC:                             2256.
    Df Residuals:                     341   BIC:                             2307.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     24.1584      7.068      3.418      0.001      10.256      38.060
    CRIM          -0.1869      0.048     -3.925      0.000      -0.281      -0.093
    ZN             0.0410      0.020      2.026      0.043       0.001       0.081
    INDUS         -0.0158      0.090     -0.177      0.860      -0.192       0.160
    CHAS           3.6109      1.217      2.967      0.003       1.217       6.005
    NOX          -20.1868      5.433     -3.716      0.000     -30.873      -9.500
    RM             5.7476      0.496     11.598      0.000       4.773       6.722
    AGE           -0.0479      0.018     -2.633      0.009      -0.084      -0.012
    DIS           -1.6023      0.286     -5.599      0.000      -2.165      -1.039
    RAD            0.3014      0.095      3.175      0.002       0.115       0.488
    TAX           -0.0110      0.005     -2.022      0.044      -0.022      -0.000
    PTRATIO       -1.1253      0.194     -5.804      0.000      -1.507      -0.744
    B              0.0152      0.004      3.948      0.000       0.008       0.023
    ==============================================================================
    Omnibus:                      184.914   Durbin-Watson:                   1.877
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1363.940
    Skew:                           2.082   Prob(JB):                    6.67e-297
    Kurtosis:                      11.668   Cond. No.                     1.49e+04
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

    r2 score: 0.7942706622921789
    mse: 15.072158695996528
    rmse: 3.8822878172537036
    mae: 3.001269274891883


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.673
    Model:                            OLS   Adj. R-squared:                  0.664
    Method:                 Least Squares   F-statistic:                     70.70
    Date:                Tue, 16 Mar 2021   Prob (F-statistic):           3.72e-77
    Time:                        21:38:49   Log-Likelihood:                 8.9419
    No. Observations:                 354   AIC:                             4.116
    Df Residuals:                     343   BIC:                             46.68
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.3254      0.293     11.360      0.000       2.750       3.901
    CRIM           -0.0144      0.002     -7.281      0.000      -0.018      -0.011
    CHAS            0.1344      0.050      2.672      0.008       0.035       0.233
    NOX            -0.9028      0.218     -4.140      0.000      -1.332      -0.474
    RM              0.2040      0.020     10.129      0.000       0.164       0.244
    DIS            -0.0494      0.011     -4.656      0.000      -0.070      -0.029
    RAD             0.0146      0.004      3.872      0.000       0.007       0.022
    TAX            -0.0006      0.000     -2.893      0.004      -0.001      -0.000
    PTRATIO        -0.0491      0.007     -6.672      0.000      -0.064      -0.035
    B               0.0007      0.000      4.642      0.000       0.000       0.001
    pow(AGE, 2) -1.953e-05   6.19e-06     -3.155      0.002   -3.17e-05   -7.35e-06
    ==============================================================================
    Omnibus:                       89.866   Durbin-Watson:                   1.902
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              573.451
    Skew:                           0.883   Prob(JB):                    3.00e-125
    Kurtosis:                       8.980   Cond. No.                     1.75e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.75e+05. This might indicate that there are
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

    r2 score: 0.8322818831273675
    mse: 12.2873776869302
    rmse: 3.505335602610711
    mae: 2.66670691478715



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
