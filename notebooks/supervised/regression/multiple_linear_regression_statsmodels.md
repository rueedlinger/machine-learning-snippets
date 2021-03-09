>**Note**: This is a generated markdown export from the Jupyter notebook file [multiple_linear_regression_statsmodels.ipynb](multiple_linear_regression_statsmodels.ipynb).

## Linear regression with statsmodels (OLS)


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools.eval_measures as eval_measures
import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import datasets, model_selection
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
    Method:                 Least Squares   F-statistic:                              550.9
    Date:                Tue, 09 Mar 2021   Prob (F-statistic):                   7.33e-220
    Time:                        00:10:59   Log-Likelihood:                         -1083.9
    No. Observations:                 354   AIC:                                      2194.
    Df Residuals:                     341   BIC:                                      2244.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1130      0.042     -2.713      0.007      -0.195      -0.031
    ZN             0.0539      0.019      2.838      0.005       0.017       0.091
    INDUS         -0.0381      0.082     -0.464      0.643      -0.199       0.123
    CHAS           2.8957      1.131      2.561      0.011       0.671       5.120
    NOX           -1.0232      4.116     -0.249      0.804      -9.120       7.073
    RM             5.3250      0.392     13.581      0.000       4.554       6.096
    AGE            0.0027      0.018      0.151      0.880      -0.032       0.037
    DIS           -1.0307      0.260     -3.963      0.000      -1.542      -0.519
    RAD            0.2582      0.087      2.958      0.003       0.086       0.430
    TAX           -0.0127      0.005     -2.467      0.014      -0.023      -0.003
    PTRATIO       -0.1599      0.142     -1.123      0.262      -0.440       0.120
    B              0.0157      0.003      4.557      0.000       0.009       0.022
    LSTAT         -0.5245      0.064     -8.208      0.000      -0.650      -0.399
    ==============================================================================
    Omnibus:                      135.280   Durbin-Watson:                   2.036
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              715.152
    Skew:                           1.527   Prob(JB):                    5.09e-156
    Kurtosis:                       9.258   Cond. No.                     8.27e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.27e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_8_0.png)
    



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
    PTRATIO    False
    B           True
    LSTAT       True
    dtype: bool




```python
print("rsquared: {}".format(result.rsquared))
print("mse: {}".format(eval_measures.mse(y_train, result.fittedvalues)))
print("rmse: {}".format(eval_measures.rmse(y_train, result.fittedvalues)))
```

    rsquared: 0.9545463039462087
    mse: 26.73304178087602
    rmse: 5.170400543562947


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.716
    Model:                            OLS   Adj. R-squared:                  0.705
    Method:                 Least Squares   F-statistic:                     65.91
    Date:                Tue, 09 Mar 2021   Prob (F-statistic):           1.39e-84
    Time:                        00:10:59   Log-Likelihood:                -1060.1
    No. Observations:                 354   AIC:                             2148.
    Df Residuals:                     340   BIC:                             2202.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         43.0115      6.149      6.995      0.000      30.916      55.107
    CRIM          -0.1220      0.039     -3.126      0.002      -0.199      -0.045
    ZN             0.0416      0.018      2.329      0.020       0.006       0.077
    INDUS          0.0240      0.077      0.311      0.756      -0.128       0.176
    CHAS           2.8767      1.059      2.717      0.007       0.794       4.959
    NOX          -19.2006      4.649     -4.130      0.000     -28.344     -10.057
    RM             2.9823      0.497      6.001      0.000       2.005       3.960
    AGE            0.0073      0.016      0.442      0.659      -0.025       0.040
    DIS           -1.5876      0.256     -6.196      0.000      -2.092      -1.084
    RAD            0.4070      0.084      4.819      0.000       0.241       0.573
    TAX           -0.0160      0.005     -3.295      0.001      -0.026      -0.006
    PTRATIO       -0.8473      0.166     -5.116      0.000      -1.173      -0.522
    B              0.0084      0.003      2.477      0.014       0.002       0.015
    LSTAT         -0.6552      0.063    -10.452      0.000      -0.778      -0.532
    ==============================================================================
    Omnibus:                      120.710   Durbin-Watson:                   2.095
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              423.559
    Skew:                           1.497   Prob(JB):                     1.06e-92
    Kurtosis:                       7.444   Cond. No.                     1.48e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.48e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_13_0.png)
    



```python
print("rsquared: {}".format(result.rsquared))
print("mse: {}".format(eval_measures.mse(y_train, result.fittedvalues)))
print("rmse: {}".format(eval_measures.rmse(y_train, result.fittedvalues)))
```

    rsquared: 0.715929580960083
    mse: 23.370244333627955
    rmse: 4.834278057127864


## Fitting models using R-style formulas
We can also fit a model with the R syntax `y ~ x_1 + x_2` and build some complexer models.


```python
dat = X_train.copy()
dat['MEDV'] = y_train
dat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>484</th>
      <td>2.37857</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.583</td>
      <td>5.871</td>
      <td>41.9</td>
      <td>3.7240</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>370.73</td>
      <td>13.34</td>
      <td>20.6</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.62739</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>5.834</td>
      <td>56.5</td>
      <td>4.4986</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>395.62</td>
      <td>8.47</td>
      <td>19.9</td>
    </tr>
    <tr>
      <th>253</th>
      <td>0.36894</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.431</td>
      <td>8.259</td>
      <td>8.4</td>
      <td>8.9067</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>396.90</td>
      <td>3.54</td>
      <td>42.8</td>
    </tr>
    <tr>
      <th>272</th>
      <td>0.11460</td>
      <td>20.0</td>
      <td>6.96</td>
      <td>0.0</td>
      <td>0.464</td>
      <td>6.538</td>
      <td>58.7</td>
      <td>3.9175</td>
      <td>3.0</td>
      <td>223.0</td>
      <td>18.6</td>
      <td>394.96</td>
      <td>7.73</td>
      <td>24.4</td>
    </tr>
    <tr>
      <th>299</th>
      <td>0.05561</td>
      <td>70.0</td>
      <td>2.24</td>
      <td>0.0</td>
      <td>0.400</td>
      <td>7.041</td>
      <td>10.0</td>
      <td>7.8278</td>
      <td>5.0</td>
      <td>358.0</td>
      <td>14.8</td>
      <td>371.58</td>
      <td>4.74</td>
      <td>29.0</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.625
    Model:                            OLS   Adj. R-squared:                  0.611
    Method:                 Least Squares   F-statistic:                     47.29
    Date:                Tue, 09 Mar 2021   Prob (F-statistic):           3.49e-65
    Time:                        00:10:59   Log-Likelihood:                -1109.4
    No. Observations:                 354   AIC:                             2245.
    Df Residuals:                     341   BIC:                             2295.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     23.8561      6.737      3.541      0.000      10.604      37.108
    CRIM          -0.1830      0.044     -4.132      0.000      -0.270      -0.096
    ZN             0.0368      0.021      1.793      0.074      -0.004       0.077
    INDUS         -0.0426      0.088     -0.482      0.630      -0.216       0.131
    CHAS           3.3538      1.214      2.762      0.006       0.965       5.742
    NOX          -20.9906      5.332     -3.937      0.000     -31.478     -10.503
    RM             5.6271      0.491     11.462      0.000       4.662       6.593
    AGE           -0.0559      0.018     -3.179      0.002      -0.091      -0.021
    DIS           -1.7835      0.293     -6.081      0.000      -2.360      -1.207
    RAD            0.3644      0.097      3.764      0.000       0.174       0.555
    TAX           -0.0142      0.006     -2.551      0.011      -0.025      -0.003
    PTRATIO       -0.9233      0.190     -4.861      0.000      -1.297      -0.550
    B              0.0153      0.004      4.005      0.000       0.008       0.023
    ==============================================================================
    Omnibus:                      186.162   Durbin-Watson:                   2.130
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1397.113
    Skew:                           2.093   Prob(JB):                    4.18e-304
    Kurtosis:                      11.786   Cond. No.                     1.45e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.45e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
fig = sm.qqplot(result.resid, fit=True, line="s")

plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_19_0.png)
    



```python
print("rsquared: {}".format(result.rsquared))
print("mse: {}".format(eval_measures.mse(y_train, result.fittedvalues)))
print("rmse: {}".format(eval_measures.rmse(y_train, result.fittedvalues)))
```

    rsquared: 0.6246606596972419
    mse: 30.8788648974591
    rmse: 5.556875461755383


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.651
    Model:                            OLS   Adj. R-squared:                  0.641
    Method:                 Least Squares   F-statistic:                     64.02
    Date:                Tue, 09 Mar 2021   Prob (F-statistic):           2.53e-72
    Time:                        00:10:59   Log-Likelihood:                 5.7805
    No. Observations:                 354   AIC:                             10.44
    Df Residuals:                     343   BIC:                             53.00
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.2315      0.284     11.379      0.000       2.673       3.790
    CRIM           -0.0130      0.002     -6.905      0.000      -0.017      -0.009
    CHAS            0.1399      0.052      2.707      0.007       0.038       0.242
    NOX            -0.9335      0.218     -4.284      0.000      -1.362      -0.505
    RM              0.2042      0.021      9.874      0.000       0.164       0.245
    DIS            -0.0548      0.011     -4.978      0.000      -0.076      -0.033
    RAD             0.0143      0.004      3.597      0.000       0.006       0.022
    TAX            -0.0006      0.000     -2.770      0.006      -0.001      -0.000
    PTRATIO        -0.0414      0.007     -5.638      0.000      -0.056      -0.027
    B               0.0008      0.000      4.663      0.000       0.000       0.001
    pow(AGE, 2) -2.163e-05   6.25e-06     -3.462      0.001   -3.39e-05   -9.34e-06
    ==============================================================================
    Omnibus:                       79.977   Durbin-Watson:                   2.125
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              508.941
    Skew:                           0.760   Prob(JB):                    3.05e-111
    Kurtosis:                       8.674   Cond. No.                     1.66e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.66e+05. This might indicate that there are
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
print("rsquared: {}".format(result.rsquared))
print("mse: {}".format(eval_measures.mse(np.log(y_train), result.fittedvalues)))
print("rmse: {}".format(eval_measures.rmse(np.log(y_train), result.fittedvalues)))
```

    rsquared: 0.6511542585545298
    mse: 0.05666858371675619
    rmse: 0.23805164086129754



```python
fig = sm.qqplot(result.resid, fit=True, line="q")
plt.show()
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_26_0.png)
    
