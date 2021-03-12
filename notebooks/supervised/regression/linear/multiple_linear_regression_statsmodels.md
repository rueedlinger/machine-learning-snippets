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
    Dep. Variable:                      y   R-squared (uncentered):                   0.964
    Model:                            OLS   Adj. R-squared (uncentered):              0.963
    Method:                 Least Squares   F-statistic:                              712.5
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):                   4.03e-238
    Time:                        13:09:32   Log-Likelihood:                         -1049.4
    No. Observations:                 354   AIC:                                      2125.
    Df Residuals:                     341   BIC:                                      2175.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0448      0.048     -0.926      0.355      -0.140       0.050
    ZN             0.0469      0.017      2.785      0.006       0.014       0.080
    INDUS          0.0305      0.072      0.421      0.674      -0.112       0.173
    CHAS           0.7708      1.066      0.723      0.470      -1.326       2.867
    NOX           -7.1236      4.210     -1.692      0.092     -15.405       1.158
    RM             6.2439      0.344     18.170      0.000       5.568       6.920
    AGE            0.0069      0.016      0.437      0.662      -0.024       0.038
    DIS           -1.0823      0.232     -4.675      0.000      -1.538      -0.627
    RAD            0.2121      0.083      2.564      0.011       0.049       0.375
    TAX           -0.0112      0.005     -2.340      0.020      -0.021      -0.002
    PTRATIO       -0.4629      0.131     -3.542      0.000      -0.720      -0.206
    B              0.0203      0.003      6.382      0.000       0.014       0.027
    LSTAT         -0.4708      0.061     -7.778      0.000      -0.590      -0.352
    ==============================================================================
    Omnibus:                      139.132   Durbin-Watson:                   2.141
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1117.805
    Skew:                           1.423   Prob(JB):                    1.87e-243
    Kurtosis:                      11.227   Cond. No.                     9.24e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 9.24e+03. This might indicate that there are
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
    CHAS       False
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

    r2 score: 0.5527790366766099
    mse: 32.0149847297891
    rmse: 5.6581785699807225
    mae: 3.9799432003630355


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.768
    Model:                            OLS   Adj. R-squared:                  0.759
    Method:                 Least Squares   F-statistic:                     86.53
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           2.46e-99
    Time:                        13:09:32   Log-Likelihood:                -1038.5
    No. Observations:                 354   AIC:                             2105.
    Df Residuals:                     340   BIC:                             2159.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         27.9102      6.017      4.639      0.000      16.075      39.745
    CRIM          -0.0734      0.047     -1.547      0.123      -0.167       0.020
    ZN             0.0467      0.016      2.857      0.005       0.015       0.079
    INDUS          0.0404      0.070      0.574      0.567      -0.098       0.179
    CHAS           0.7897      1.035      0.763      0.446      -1.246       2.826
    NOX          -18.1496      4.730     -3.837      0.000     -27.453      -8.846
    RM             4.5934      0.488      9.416      0.000       3.634       5.553
    AGE            0.0126      0.015      0.816      0.415      -0.018       0.043
    DIS           -1.4712      0.240     -6.131      0.000      -1.943      -0.999
    RAD            0.3155      0.083      3.784      0.000       0.151       0.479
    TAX           -0.0130      0.005     -2.783      0.006      -0.022      -0.004
    PTRATIO       -0.8690      0.154     -5.636      0.000      -1.172      -0.566
    B              0.0149      0.003      4.512      0.000       0.008       0.021
    LSTAT         -0.5601      0.062     -9.054      0.000      -0.682      -0.438
    ==============================================================================
    Omnibus:                      121.840   Durbin-Watson:                   2.110
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              602.459
    Skew:                           1.375   Prob(JB):                    1.51e-131
    Kurtosis:                       8.769   Cond. No.                     1.52e+04
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

    r2 score: 0.6169930338526008
    mse: 27.41812924307191
    rmse: 5.236232351898826
    mae: 3.856484487052427


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
      <th>338</th>
      <td>0.03306</td>
      <td>0.0</td>
      <td>5.19</td>
      <td>0.0</td>
      <td>0.515</td>
      <td>6.059</td>
      <td>37.3</td>
      <td>4.8122</td>
      <td>5.0</td>
      <td>224.0</td>
      <td>20.2</td>
      <td>396.14</td>
      <td>8.51</td>
      <td>20.6</td>
    </tr>
    <tr>
      <th>309</th>
      <td>0.34940</td>
      <td>0.0</td>
      <td>9.90</td>
      <td>0.0</td>
      <td>0.544</td>
      <td>5.972</td>
      <td>76.7</td>
      <td>3.1025</td>
      <td>4.0</td>
      <td>304.0</td>
      <td>18.4</td>
      <td>396.24</td>
      <td>9.97</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>134</th>
      <td>0.97617</td>
      <td>0.0</td>
      <td>21.89</td>
      <td>0.0</td>
      <td>0.624</td>
      <td>5.757</td>
      <td>98.4</td>
      <td>2.3460</td>
      <td>4.0</td>
      <td>437.0</td>
      <td>21.2</td>
      <td>262.76</td>
      <td>17.31</td>
      <td>15.6</td>
    </tr>
    <tr>
      <th>492</th>
      <td>0.11132</td>
      <td>0.0</td>
      <td>27.74</td>
      <td>0.0</td>
      <td>0.609</td>
      <td>5.983</td>
      <td>83.5</td>
      <td>2.1099</td>
      <td>4.0</td>
      <td>711.0</td>
      <td>20.1</td>
      <td>396.90</td>
      <td>13.35</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>410</th>
      <td>51.13580</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.597</td>
      <td>5.757</td>
      <td>100.0</td>
      <td>1.4130</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>2.60</td>
      <td>10.11</td>
      <td>15.0</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.712
    Model:                            OLS   Adj. R-squared:                  0.702
    Method:                 Least Squares   F-statistic:                     70.23
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           1.69e-84
    Time:                        13:09:33   Log-Likelihood:                -1076.7
    No. Observations:                 354   AIC:                             2179.
    Df Residuals:                     341   BIC:                             2230.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     10.9646      6.361      1.724      0.086      -1.548      23.477
    CRIM          -0.1440      0.052     -2.765      0.006      -0.246      -0.042
    ZN             0.0360      0.018      1.981      0.048       0.000       0.072
    INDUS         -0.0008      0.078     -0.010      0.992      -0.155       0.153
    CHAS           1.0050      1.151      0.873      0.383      -1.259       3.269
    NOX          -17.9347      5.262     -3.409      0.001     -28.284      -7.585
    RM             6.9838      0.456     15.303      0.000       6.086       7.881
    AGE           -0.0399      0.016     -2.516      0.012      -0.071      -0.009
    DIS           -1.5094      0.267     -5.655      0.000      -2.034      -0.984
    RAD            0.2636      0.093      2.848      0.005       0.082       0.446
    TAX           -0.0116      0.005     -2.248      0.025      -0.022      -0.001
    PTRATIO       -1.0117      0.171     -5.929      0.000      -1.347      -0.676
    B              0.0202      0.004      5.593      0.000       0.013       0.027
    ==============================================================================
    Omnibus:                      211.363   Durbin-Watson:                   2.102
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2695.268
    Skew:                           2.243   Prob(JB):                         0.00
    Kurtosis:                      15.752   Cond. No.                     1.48e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.48e+04. This might indicate that there are
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

    r2 score: 0.5569114556903292
    mse: 31.71915930461634
    rmse: 5.63197650071592
    mae: 3.806457031677857


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.730
    Model:                            OLS   Adj. R-squared:                  0.722
    Method:                 Least Squares   F-statistic:                     92.67
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           3.61e-91
    Time:                        13:09:33   Log-Likelihood:                 40.765
    No. Observations:                 354   AIC:                            -59.53
    Df Residuals:                     343   BIC:                            -16.97
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       2.8216      0.270     10.453      0.000       2.291       3.352
    CRIM           -0.0152      0.002     -6.917      0.000      -0.020      -0.011
    CHAS            0.0624      0.049      1.284      0.200      -0.033       0.158
    NOX            -0.8334      0.218     -3.829      0.000      -1.261      -0.405
    RM              0.2454      0.019     12.995      0.000       0.208       0.283
    DIS            -0.0461      0.009     -4.870      0.000      -0.065      -0.027
    RAD             0.0091      0.004      2.411      0.016       0.002       0.017
    TAX            -0.0003      0.000     -1.654      0.099      -0.001    6.14e-05
    PTRATIO        -0.0438      0.007     -6.488      0.000      -0.057      -0.031
    B               0.0009      0.000      5.691      0.000       0.001       0.001
    pow(AGE, 2) -1.736e-05   5.71e-06     -3.041      0.003   -2.86e-05   -6.13e-06
    ==============================================================================
    Omnibus:                      104.840   Durbin-Watson:                   1.982
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              996.307
    Skew:                           0.935   Prob(JB):                    4.52e-217
    Kurtosis:                      11.003   Cond. No.                     1.70e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.7e+05. This might indicate that there are
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

    r2 score: 0.6708346169116406
    mse: 23.563798608270314
    rmse: 4.8542557213511435
    mae: 2.940830939177958



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
