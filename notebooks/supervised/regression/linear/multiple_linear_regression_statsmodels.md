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
    Model:                            OLS   Adj. R-squared (uncentered):              0.959
    Method:                 Least Squares   F-statistic:                              639.2
    Date:                Mon, 22 Mar 2021   Prob (F-statistic):                   2.14e-230
    Time:                        15:17:07   Log-Likelihood:                         -1059.6
    No. Observations:                 354   AIC:                                      2145.
    Df Residuals:                     341   BIC:                                      2196.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0875      0.040     -2.196      0.029      -0.166      -0.009
    ZN             0.0486      0.019      2.592      0.010       0.012       0.085
    INDUS         -0.0222      0.081     -0.276      0.783      -0.181       0.136
    CHAS           3.9935      1.091      3.661      0.000       1.848       6.139
    NOX            1.3155      4.068      0.323      0.747      -6.685       9.316
    RM             5.7816      0.355     16.303      0.000       5.084       6.479
    AGE           -0.0036      0.017     -0.215      0.830      -0.036       0.029
    DIS           -0.8258      0.237     -3.479      0.001      -1.293      -0.359
    RAD            0.1497      0.078      1.917      0.056      -0.004       0.303
    TAX           -0.0075      0.005     -1.573      0.117      -0.017       0.002
    PTRATIO       -0.4494      0.129     -3.474      0.001      -0.704      -0.195
    B              0.0134      0.003      4.149      0.000       0.007       0.020
    LSTAT         -0.4975      0.059     -8.489      0.000      -0.613      -0.382
    ==============================================================================
    Omnibus:                      141.459   Durbin-Watson:                   2.054
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              750.113
    Skew:                           1.609   Prob(JB):                    1.30e-163
    Kurtosis:                       9.364   Cond. No.                     8.69e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.69e+03. This might indicate that there are
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

    r2 score: 0.679993896286366
    mse: 27.36332302897757
    rmse: 5.230996370575836
    mae: 3.4445495767595267


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.750
    Model:                            OLS   Adj. R-squared:                  0.741
    Method:                 Least Squares   F-statistic:                     78.64
    Date:                Mon, 22 Mar 2021   Prob (F-statistic):           4.98e-94
    Time:                        15:17:07   Log-Likelihood:                -1040.8
    No. Observations:                 354   AIC:                             2110.
    Df Residuals:                     340   BIC:                             2164.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         36.7873      5.955      6.177      0.000      25.073      48.501
    CRIM          -0.1090      0.038     -2.867      0.004      -0.184      -0.034
    ZN             0.0504      0.018      2.831      0.005       0.015       0.085
    INDUS         -0.0029      0.077     -0.038      0.970      -0.154       0.148
    CHAS           3.6980      1.037      3.566      0.000       1.658       5.738
    NOX          -14.4445      4.629     -3.120      0.002     -23.550      -5.339
    RM             3.6358      0.484      7.515      0.000       2.684       4.587
    AGE            0.0068      0.016      0.429      0.668      -0.024       0.038
    DIS           -1.3878      0.243     -5.709      0.000      -1.866      -0.910
    RAD            0.2885      0.078      3.722      0.000       0.136       0.441
    TAX           -0.0111      0.005     -2.415      0.016      -0.020      -0.002
    PTRATIO       -0.9895      0.151     -6.563      0.000      -1.286      -0.693
    B              0.0080      0.003      2.497      0.013       0.002       0.014
    LSTAT         -0.5909      0.058    -10.247      0.000      -0.704      -0.478
    ==============================================================================
    Omnibus:                      115.149   Durbin-Watson:                   2.070
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              404.682
    Skew:                           1.421   Prob(JB):                     1.33e-88
    Kurtosis:                       7.400   Cond. No.                     1.51e+04
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

    r2 score: 0.7083989633563967
    mse: 24.934441151797586
    rmse: 4.9934398115725385
    mae: 3.4048545421334255


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
      <th>188</th>
      <td>0.12579</td>
      <td>45.0</td>
      <td>3.44</td>
      <td>0.0</td>
      <td>0.437</td>
      <td>6.556</td>
      <td>29.1</td>
      <td>4.5667</td>
      <td>5.0</td>
      <td>398.0</td>
      <td>15.2</td>
      <td>382.84</td>
      <td>4.56</td>
      <td>29.8</td>
    </tr>
    <tr>
      <th>363</th>
      <td>4.22239</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>1.0</td>
      <td>0.770</td>
      <td>5.803</td>
      <td>89.0</td>
      <td>1.9047</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>353.04</td>
      <td>14.64</td>
      <td>16.8</td>
    </tr>
    <tr>
      <th>349</th>
      <td>0.02899</td>
      <td>40.0</td>
      <td>1.25</td>
      <td>0.0</td>
      <td>0.429</td>
      <td>6.939</td>
      <td>34.5</td>
      <td>8.7921</td>
      <td>1.0</td>
      <td>335.0</td>
      <td>19.7</td>
      <td>389.85</td>
      <td>5.89</td>
      <td>26.6</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.12083</td>
      <td>0.0</td>
      <td>2.89</td>
      <td>0.0</td>
      <td>0.445</td>
      <td>8.069</td>
      <td>76.0</td>
      <td>3.4952</td>
      <td>2.0</td>
      <td>276.0</td>
      <td>18.0</td>
      <td>396.90</td>
      <td>4.21</td>
      <td>38.7</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.18836</td>
      <td>0.0</td>
      <td>6.91</td>
      <td>0.0</td>
      <td>0.448</td>
      <td>5.786</td>
      <td>33.3</td>
      <td>5.1004</td>
      <td>3.0</td>
      <td>233.0</td>
      <td>17.9</td>
      <td>396.90</td>
      <td>14.15</td>
      <td>20.0</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.673
    Model:                            OLS   Adj. R-squared:                  0.662
    Method:                 Least Squares   F-statistic:                     58.58
    Date:                Mon, 22 Mar 2021   Prob (F-statistic):           2.60e-75
    Time:                        15:17:07   Log-Likelihood:                -1088.4
    No. Observations:                 354   AIC:                             2203.
    Df Residuals:                     341   BIC:                             2253.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     20.7817      6.565      3.166      0.002       7.869      33.694
    CRIM          -0.1452      0.043     -3.357      0.001      -0.230      -0.060
    ZN             0.0384      0.020      1.893      0.059      -0.002       0.078
    INDUS         -0.0117      0.088     -0.134      0.894      -0.184       0.161
    CHAS           4.1996      1.183      3.549      0.000       1.872       6.527
    NOX          -18.3850      5.270     -3.489      0.001     -28.750      -8.020
    RM             5.9829      0.487     12.290      0.000       5.025       6.940
    AGE           -0.0398      0.017     -2.297      0.022      -0.074      -0.006
    DIS           -1.3270      0.278     -4.780      0.000      -1.873      -0.781
    RAD            0.2580      0.088      2.917      0.004       0.084       0.432
    TAX           -0.0111      0.005     -2.123      0.034      -0.021      -0.001
    PTRATIO       -1.1029      0.172     -6.421      0.000      -1.441      -0.765
    B              0.0129      0.004      3.585      0.000       0.006       0.020
    ==============================================================================
    Omnibus:                      196.912   Durbin-Watson:                   1.967
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1853.290
    Skew:                           2.154   Prob(JB):                         0.00
    Kurtosis:                      13.348   Cond. No.                     1.49e+04
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

    r2 score: 0.7004866769448531
    mse: 25.61101089988778
    rmse: 5.060732249377335
    mae: 3.442966592564623


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.676
    Model:                            OLS   Adj. R-squared:                  0.666
    Method:                 Least Squares   F-statistic:                     71.53
    Date:                Mon, 22 Mar 2021   Prob (F-statistic):           9.70e-78
    Time:                        15:17:08   Log-Likelihood:                 20.731
    No. Observations:                 354   AIC:                            -19.46
    Df Residuals:                     343   BIC:                             23.10
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.1191      0.287     10.869      0.000       2.555       3.684
    CRIM           -0.0097      0.002     -5.165      0.000      -0.013      -0.006
    CHAS            0.1727      0.051      3.378      0.001       0.072       0.273
    NOX            -0.8486      0.224     -3.793      0.000      -1.289      -0.409
    RM              0.2122      0.021     10.344      0.000       0.172       0.253
    DIS            -0.0362      0.010     -3.645      0.000      -0.056      -0.017
    RAD             0.0092      0.004      2.562      0.011       0.002       0.016
    TAX            -0.0004      0.000     -2.229      0.026      -0.001   -4.98e-05
    PTRATIO        -0.0469      0.007     -6.795      0.000      -0.061      -0.033
    B               0.0007      0.000      4.635      0.000       0.000       0.001
    pow(AGE, 2) -1.612e-05   6.26e-06     -2.576      0.010   -2.84e-05   -3.81e-06
    ==============================================================================
    Omnibus:                       81.247   Durbin-Watson:                   1.942
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              689.822
    Skew:                           0.675   Prob(JB):                    1.61e-150
    Kurtosis:                       9.704   Cond. No.                     1.76e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.76e+05. This might indicate that there are
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

    r2 score: 0.7078742524169689
    mse: 24.979308530156253
    rmse: 4.997930424701433
    mae: 3.229202014370454



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
