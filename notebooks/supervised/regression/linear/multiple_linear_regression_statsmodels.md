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
    Dep. Variable:                      y   R-squared (uncentered):                   0.957
    Model:                            OLS   Adj. R-squared (uncentered):              0.956
    Method:                 Least Squares   F-statistic:                              585.9
    Date:                Fri, 26 Mar 2021   Prob (F-statistic):                   3.23e-224
    Time:                        10:35:57   Log-Likelihood:                         -1071.1
    No. Observations:                 354   AIC:                                      2168.
    Df Residuals:                     341   BIC:                                      2219.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0944      0.040     -2.346      0.020      -0.173      -0.015
    ZN             0.0489      0.019      2.577      0.010       0.012       0.086
    INDUS          0.0099      0.079      0.125      0.900      -0.145       0.165
    CHAS           2.1543      1.209      1.782      0.076      -0.223       4.532
    NOX           -3.1645      4.186     -0.756      0.450     -11.398       5.069
    RM             5.9229      0.366     16.183      0.000       5.203       6.643
    AGE           -0.0049      0.016     -0.301      0.764      -0.037       0.027
    DIS           -1.0352      0.247     -4.198      0.000      -1.520      -0.550
    RAD            0.1708      0.086      1.976      0.049       0.001       0.341
    TAX           -0.0094      0.005     -1.876      0.062      -0.019       0.000
    PTRATIO       -0.3862      0.133     -2.904      0.004      -0.648      -0.125
    B              0.0162      0.003      4.875      0.000       0.010       0.023
    LSTAT         -0.4435      0.060     -7.407      0.000      -0.561      -0.326
    ==============================================================================
    Omnibus:                      148.438   Durbin-Watson:                   1.945
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              769.501
    Skew:                           1.713   Prob(JB):                    8.03e-168
    Kurtosis:                       9.358   Cond. No.                     8.65e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.65e+03. This might indicate that there are
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
    CHAS       False
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

    r2 score: 0.7282349202025054
    mse: 22.761575100720137
    rmse: 4.770909253037637
    mae: 3.0531380018410186


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.737
    Model:                            OLS   Adj. R-squared:                  0.727
    Method:                 Least Squares   F-statistic:                     73.18
    Date:                Fri, 26 Mar 2021   Prob (F-statistic):           3.99e-90
    Time:                        10:35:57   Log-Likelihood:                -1051.3
    No. Observations:                 354   AIC:                             2131.
    Df Residuals:                     340   BIC:                             2185.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         40.0457      6.315      6.341      0.000      27.624      52.468
    CRIM          -0.1092      0.038     -2.862      0.004      -0.184      -0.034
    ZN             0.0483      0.018      2.687      0.008       0.013       0.084
    INDUS          0.0350      0.075      0.469      0.640      -0.112       0.182
    CHAS           1.7514      1.147      1.528      0.128      -0.504       4.007
    NOX          -19.4822      4.726     -4.122      0.000     -28.778     -10.186
    RM             3.6205      0.502      7.212      0.000       2.633       4.608
    AGE         -5.14e-07      0.015  -3.36e-05      1.000      -0.030       0.030
    DIS           -1.6236      0.251     -6.461      0.000      -2.118      -1.129
    RAD            0.3340      0.086      3.892      0.000       0.165       0.503
    TAX           -0.0132      0.005     -2.760      0.006      -0.023      -0.004
    PTRATIO       -1.0094      0.160     -6.318      0.000      -1.324      -0.695
    B              0.0107      0.003      3.289      0.001       0.004       0.017
    LSTAT         -0.5464      0.059     -9.264      0.000      -0.662      -0.430
    ==============================================================================
    Omnibus:                      121.805   Durbin-Watson:                   1.989
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              435.075
    Skew:                           1.505   Prob(JB):                     3.35e-95
    Kurtosis:                       7.521   Cond. No.                     1.54e+04
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

    r2 score: 0.7437718639995317
    mse: 21.460284613600884
    rmse: 4.632524647921572
    mae: 3.2574639509766166


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
      <th>446</th>
      <td>6.28807</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>6.341</td>
      <td>96.4</td>
      <td>2.0720</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>318.01</td>
      <td>17.79</td>
      <td>14.9</td>
    </tr>
    <tr>
      <th>176</th>
      <td>0.07022</td>
      <td>0.0</td>
      <td>4.05</td>
      <td>0.0</td>
      <td>0.510</td>
      <td>6.020</td>
      <td>47.2</td>
      <td>3.5549</td>
      <td>5.0</td>
      <td>296.0</td>
      <td>16.6</td>
      <td>393.23</td>
      <td>10.11</td>
      <td>23.2</td>
    </tr>
    <tr>
      <th>172</th>
      <td>0.13914</td>
      <td>0.0</td>
      <td>4.05</td>
      <td>0.0</td>
      <td>0.510</td>
      <td>5.572</td>
      <td>88.5</td>
      <td>2.5961</td>
      <td>5.0</td>
      <td>296.0</td>
      <td>16.6</td>
      <td>396.90</td>
      <td>14.69</td>
      <td>23.1</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.05302</td>
      <td>0.0</td>
      <td>3.41</td>
      <td>0.0</td>
      <td>0.489</td>
      <td>7.079</td>
      <td>63.1</td>
      <td>3.4145</td>
      <td>2.0</td>
      <td>270.0</td>
      <td>17.8</td>
      <td>396.06</td>
      <td>5.70</td>
      <td>28.7</td>
    </tr>
    <tr>
      <th>275</th>
      <td>0.09604</td>
      <td>40.0</td>
      <td>6.41</td>
      <td>0.0</td>
      <td>0.447</td>
      <td>6.854</td>
      <td>42.8</td>
      <td>4.2673</td>
      <td>4.0</td>
      <td>254.0</td>
      <td>17.6</td>
      <td>396.90</td>
      <td>2.98</td>
      <td>32.0</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.670
    Model:                            OLS   Adj. R-squared:                  0.659
    Method:                 Least Squares   F-statistic:                     57.76
    Date:                Fri, 26 Mar 2021   Prob (F-statistic):           1.27e-74
    Time:                        10:35:57   Log-Likelihood:                -1091.2
    No. Observations:                 354   AIC:                             2208.
    Df Residuals:                     341   BIC:                             2259.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     23.9519      6.785      3.530      0.000      10.606      37.297
    CRIM          -0.1667      0.042     -3.963      0.000      -0.249      -0.084
    ZN             0.0463      0.020      2.305      0.022       0.007       0.086
    INDUS         -0.0329      0.083     -0.397      0.692      -0.196       0.130
    CHAS           2.8091      1.275      2.203      0.028       0.302       5.317
    NOX          -22.2469      5.271     -4.221      0.000     -32.614     -11.880
    RM             6.0289      0.480     12.564      0.000       5.085       6.973
    AGE           -0.0477      0.016     -2.961      0.003      -0.079      -0.016
    DIS           -1.7593      0.280     -6.276      0.000      -2.311      -1.208
    RAD            0.3049      0.096      3.182      0.002       0.116       0.493
    TAX           -0.0131      0.005     -2.443      0.015      -0.024      -0.003
    PTRATIO       -1.0293      0.179     -5.765      0.000      -1.380      -0.678
    B              0.0130      0.004      3.578      0.000       0.006       0.020
    ==============================================================================
    Omnibus:                      202.636   Durbin-Watson:                   2.055
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1863.619
    Skew:                           2.249   Prob(JB):                         0.00
    Kurtosis:                      13.301   Cond. No.                     1.51e+04
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

    r2 score: 0.7121848687609877
    mse: 24.105840712507877
    rmse: 4.909769924600121
    mae: 3.374624766701121


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.685
    Model:                            OLS   Adj. R-squared:                  0.676
    Method:                 Least Squares   F-statistic:                     74.76
    Date:                Fri, 26 Mar 2021   Prob (F-statistic):           5.97e-80
    Time:                        10:35:57   Log-Likelihood:                 15.586
    No. Observations:                 354   AIC:                            -9.171
    Df Residuals:                     343   BIC:                             33.39
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.3007      0.297     11.130      0.000       2.717       3.884
    CRIM           -0.0120      0.002     -6.538      0.000      -0.016      -0.008
    CHAS            0.1263      0.056      2.273      0.024       0.017       0.236
    NOX            -1.0551      0.225     -4.684      0.000      -1.498      -0.612
    RM              0.2236      0.021     10.898      0.000       0.183       0.264
    DIS            -0.0527      0.010     -5.141      0.000      -0.073      -0.033
    RAD             0.0121      0.004      2.968      0.003       0.004       0.020
    TAX            -0.0006      0.000     -2.595      0.010      -0.001      -0.000
    PTRATIO        -0.0458      0.007     -6.241      0.000      -0.060      -0.031
    B               0.0006      0.000      3.642      0.000       0.000       0.001
    pow(AGE, 2) -1.896e-05   5.97e-06     -3.175      0.002   -3.07e-05   -7.21e-06
    ==============================================================================
    Omnibus:                       95.425   Durbin-Watson:                   2.106
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              715.004
    Skew:                           0.898   Prob(JB):                    5.48e-156
    Kurtosis:                       9.727   Cond. No.                     1.79e+05
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

    r2 score: 0.7333871723546148
    mse: 22.33005029118437
    rmse: 4.725468261578356
    mae: 3.0573533832621536



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
