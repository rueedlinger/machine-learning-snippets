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
    Dep. Variable:                      y   R-squared (uncentered):                   0.960
    Model:                            OLS   Adj. R-squared (uncentered):              0.959
    Method:                 Least Squares   F-statistic:                              637.2
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):                   3.62e-230
    Time:                        19:21:08   Log-Likelihood:                         -1059.8
    No. Observations:                 354   AIC:                                      2146.
    Df Residuals:                     341   BIC:                                      2196.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1157      0.036     -3.175      0.002      -0.187      -0.044
    ZN             0.0529      0.016      3.332      0.001       0.022       0.084
    INDUS         -0.0579      0.076     -0.758      0.449      -0.208       0.092
    CHAS           2.9838      1.099      2.715      0.007       0.822       5.145
    NOX            0.7924      4.016      0.197      0.844      -7.108       8.692
    RM             5.5754      0.371     15.044      0.000       4.846       6.304
    AGE           -0.0193      0.016     -1.203      0.230      -0.051       0.012
    DIS           -1.0791      0.228     -4.730      0.000      -1.528      -0.630
    RAD            0.1606      0.079      2.031      0.043       0.005       0.316
    TAX           -0.0081      0.004     -1.811      0.071      -0.017       0.001
    PTRATIO       -0.2525      0.133     -1.896      0.059      -0.514       0.009
    B              0.0139      0.003      4.489      0.000       0.008       0.020
    LSTAT         -0.4717      0.062     -7.611      0.000      -0.594      -0.350
    ==============================================================================
    Omnibus:                      174.816   Durbin-Watson:                   2.083
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1220.288
    Skew:                           1.960   Prob(JB):                    1.04e-265
    Kurtosis:                      11.208   Cond. No.                     8.61e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.61e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")
plt.show()
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

    r2 score: 0.7091650272644867
    mse: 26.992240748544713
    rmse: 5.195405734737636
    mae: 3.513355292155093


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.738
    Model:                            OLS   Adj. R-squared:                  0.728
    Method:                 Least Squares   F-statistic:                     73.80
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           1.42e-90
    Time:                        19:21:08   Log-Likelihood:                -1042.4
    No. Observations:                 354   AIC:                             2113.
    Df Residuals:                     340   BIC:                             2167.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         36.4258      6.142      5.931      0.000      24.345      48.506
    CRIM          -0.1284      0.035     -3.689      0.000      -0.197      -0.060
    ZN             0.0484      0.015      3.188      0.002       0.019       0.078
    INDUS         -0.0291      0.073     -0.399      0.690      -0.173       0.114
    CHAS           2.7958      1.048      2.667      0.008       0.734       4.857
    NOX          -15.1707      4.680     -3.241      0.001     -24.377      -5.964
    RM             3.5631      0.490      7.274      0.000       2.600       4.527
    AGE           -0.0109      0.015     -0.710      0.478      -0.041       0.019
    DIS           -1.5637      0.232     -6.731      0.000      -2.021      -1.107
    RAD            0.2917      0.079      3.712      0.000       0.137       0.446
    TAX           -0.0105      0.004     -2.448      0.015      -0.019      -0.002
    PTRATIO       -0.8465      0.162     -5.236      0.000      -1.165      -0.529
    B              0.0085      0.003      2.740      0.006       0.002       0.015
    LSTAT         -0.5626      0.061     -9.217      0.000      -0.683      -0.443
    ==============================================================================
    Omnibus:                      151.845   Durbin-Watson:                   2.069
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              781.221
    Skew:                           1.764   Prob(JB):                    2.29e-170
    Kurtosis:                       9.365   Cond. No.                     1.56e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.56e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_14_0.png)
    



```python
predicted = result.predict(sm.add_constant(X_test))

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.7386167562518551
    mse: 24.258841282137045
    rmse: 4.925326515281707
    mae: 3.464233554206377


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
      <th>158</th>
      <td>1.34284</td>
      <td>0.0</td>
      <td>19.58</td>
      <td>0.0</td>
      <td>0.605</td>
      <td>6.066</td>
      <td>100.0</td>
      <td>1.7573</td>
      <td>5.0</td>
      <td>403.0</td>
      <td>14.7</td>
      <td>353.89</td>
      <td>6.43</td>
      <td>24.3</td>
    </tr>
    <tr>
      <th>421</th>
      <td>7.02259</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.718</td>
      <td>6.006</td>
      <td>95.3</td>
      <td>1.8746</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>319.98</td>
      <td>15.70</td>
      <td>14.2</td>
    </tr>
    <tr>
      <th>65</th>
      <td>0.03584</td>
      <td>80.0</td>
      <td>3.37</td>
      <td>0.0</td>
      <td>0.398</td>
      <td>6.290</td>
      <td>17.8</td>
      <td>6.6115</td>
      <td>4.0</td>
      <td>337.0</td>
      <td>16.1</td>
      <td>396.90</td>
      <td>4.67</td>
      <td>23.5</td>
    </tr>
    <tr>
      <th>381</th>
      <td>15.87440</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.671</td>
      <td>6.545</td>
      <td>99.1</td>
      <td>1.5192</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>21.08</td>
      <td>10.9</td>
    </tr>
    <tr>
      <th>376</th>
      <td>15.28800</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.671</td>
      <td>6.649</td>
      <td>93.3</td>
      <td>1.3449</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>363.02</td>
      <td>23.24</td>
      <td>13.9</td>
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
    Model:                            OLS   Adj. R-squared:                  0.661
    Method:                 Least Squares   F-statistic:                     58.47
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           3.21e-75
    Time:                        19:21:08   Log-Likelihood:                -1081.9
    No. Observations:                 354   AIC:                             2190.
    Df Residuals:                     341   BIC:                             2240.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     22.2062      6.636      3.346      0.001       9.153      35.260
    CRIM          -0.1686      0.039     -4.376      0.000      -0.244      -0.093
    ZN             0.0474      0.017      2.802      0.005       0.014       0.081
    INDUS         -0.1120      0.081     -1.386      0.167      -0.271       0.047
    CHAS           3.7715      1.164      3.240      0.001       1.482       6.061
    NOX          -18.7651      5.207     -3.604      0.000     -29.007      -8.524
    RM             5.5816      0.489     11.411      0.000       4.619       6.544
    AGE           -0.0561      0.016     -3.459      0.001      -0.088      -0.024
    DIS           -1.7314      0.259     -6.696      0.000      -2.240      -1.223
    RAD            0.2838      0.088      3.236      0.001       0.111       0.456
    TAX           -0.0107      0.005     -2.225      0.027      -0.020      -0.001
    PTRATIO       -0.8620      0.180     -4.776      0.000      -1.217      -0.507
    B              0.0132      0.003      3.853      0.000       0.006       0.020
    ==============================================================================
    Omnibus:                      218.417   Durbin-Watson:                   1.960
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2428.453
    Skew:                           2.407   Prob(JB):                         0.00
    Kurtosis:                      14.894   Cond. No.                     1.54e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.54e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")

plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_20_0.png)
    



```python
predicted = result.predict(X_test)

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.6960751129666
    mse: 28.207108805102997
    rmse: 5.311036509486919
    mae: 3.7103241350955973


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.690
    Model:                            OLS   Adj. R-squared:                  0.681
    Method:                 Least Squares   F-statistic:                     76.39
    Date:                Wed, 10 Mar 2021   Prob (F-statistic):           4.77e-81
    Time:                        19:21:08   Log-Likelihood:                 31.064
    No. Observations:                 354   AIC:                            -40.13
    Df Residuals:                     343   BIC:                             2.434
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.2140      0.286     11.231      0.000       2.651       3.777
    CRIM           -0.0118      0.002     -7.106      0.000      -0.015      -0.009
    CHAS            0.1504      0.050      3.020      0.003       0.052       0.248
    NOX            -0.9612      0.218     -4.415      0.000      -1.389      -0.533
    RM              0.2066      0.020     10.234      0.000       0.167       0.246
    DIS            -0.0505      0.010     -5.167      0.000      -0.070      -0.031
    RAD             0.0120      0.004      3.233      0.001       0.005       0.019
    TAX            -0.0005      0.000     -2.586      0.010      -0.001      -0.000
    PTRATIO        -0.0406      0.007     -5.684      0.000      -0.055      -0.027
    B               0.0007      0.000      4.727      0.000       0.000       0.001
    pow(AGE, 2) -2.314e-05   5.87e-06     -3.942      0.000   -3.47e-05   -1.16e-05
    ==============================================================================
    Omnibus:                      113.440   Durbin-Watson:                   1.808
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              835.361
    Skew:                           1.128   Prob(JB):                    4.01e-182
    Kurtosis:                      10.179   Cond. No.                     1.76e+05
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

    r2 score: 0.7143037845830851
    mse: 26.51531538641471
    rmse: 5.149302417455661
    mae: 3.331732508356684



```python
fig = sm.qqplot(result.resid, fit=True, line="q")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
