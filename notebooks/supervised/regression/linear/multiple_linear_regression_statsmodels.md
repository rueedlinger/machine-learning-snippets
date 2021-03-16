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
    Dep. Variable:                      y   R-squared (uncentered):                   0.957
    Model:                            OLS   Adj. R-squared (uncentered):              0.956
    Method:                 Least Squares   F-statistic:                              587.1
    Date:                Tue, 16 Mar 2021   Prob (F-statistic):                   2.32e-224
    Time:                        08:54:04   Log-Likelihood:                         -1079.6
    No. Observations:                 354   AIC:                                      2185.
    Df Residuals:                     341   BIC:                                      2236.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0135      0.054     -0.250      0.803      -0.120       0.093
    ZN             0.0546      0.017      3.165      0.002       0.021       0.089
    INDUS          0.0530      0.083      0.635      0.526      -0.111       0.217
    CHAS           3.8987      1.146      3.401      0.001       1.644       6.153
    NOX           -6.6946      4.433     -1.510      0.132     -15.414       2.025
    RM             5.9556      0.398     14.982      0.000       5.174       6.737
    AGE           -0.0129      0.017     -0.777      0.438      -0.046       0.020
    DIS           -1.0490      0.235     -4.468      0.000      -1.511      -0.587
    RAD            0.1365      0.083      1.644      0.101      -0.027       0.300
    TAX           -0.0099      0.005     -2.013      0.045      -0.020      -0.000
    PTRATIO       -0.2440      0.144     -1.696      0.091      -0.527       0.039
    B              0.0146      0.003      4.336      0.000       0.008       0.021
    LSTAT         -0.4388      0.064     -6.890      0.000      -0.564      -0.314
    ==============================================================================
    Omnibus:                      150.474   Durbin-Watson:                   2.032
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1028.388
    Skew:                           1.631   Prob(JB):                    4.88e-224
    Kurtosis:                      10.686   Cond. No.                     8.94e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.94e+03. This might indicate that there are
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
    TAX         True
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

    r2 score: 0.6988044670545104
    mse: 21.620819557212233
    rmse: 4.649819303716246
    mae: 3.3490604018313412


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.735
    Model:                            OLS   Adj. R-squared:                  0.725
    Method:                 Least Squares   F-statistic:                     72.42
    Date:                Tue, 16 Mar 2021   Prob (F-statistic):           1.47e-89
    Time:                        08:54:04   Log-Likelihood:                -1063.1
    No. Observations:                 354   AIC:                             2154.
    Df Residuals:                     340   BIC:                             2208.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         37.0825      6.436      5.762      0.000      24.424      49.741
    CRIM          -0.0501      0.052     -0.960      0.338      -0.153       0.053
    ZN             0.0489      0.017      2.960      0.003       0.016       0.081
    INDUS          0.1058      0.080      1.318      0.188      -0.052       0.264
    CHAS           3.4584      1.098      3.149      0.002       1.298       5.619
    NOX          -22.5089      5.049     -4.458      0.000     -32.439     -12.578
    RM             3.8442      0.528      7.282      0.000       2.806       4.883
    AGE           -0.0071      0.016     -0.447      0.655      -0.038       0.024
    DIS           -1.5050      0.238     -6.325      0.000      -1.973      -1.037
    RAD            0.2926      0.084      3.488      0.001       0.128       0.458
    TAX           -0.0134      0.005     -2.834      0.005      -0.023      -0.004
    PTRATIO       -0.8070      0.169     -4.785      0.000      -1.139      -0.475
    B              0.0082      0.003      2.400      0.017       0.001       0.015
    LSTAT         -0.5484      0.064     -8.598      0.000      -0.674      -0.423
    ==============================================================================
    Omnibus:                      134.350   Durbin-Watson:                   2.044
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              580.198
    Skew:                           1.596   Prob(JB):                    1.03e-126
    Kurtosis:                       8.399   Cond. No.                     1.53e+04
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

    r2 score: 0.7307413485722212
    mse: 19.328283722560734
    rmse: 4.396394400251271
    mae: 3.3453233498570687


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
      <th>465</th>
      <td>3.16360</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.655</td>
      <td>5.759</td>
      <td>48.2</td>
      <td>3.0665</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>334.40</td>
      <td>14.13</td>
      <td>19.9</td>
    </tr>
    <tr>
      <th>469</th>
      <td>13.07510</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.580</td>
      <td>5.713</td>
      <td>56.7</td>
      <td>2.8237</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>14.76</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>217</th>
      <td>0.07013</td>
      <td>0.0</td>
      <td>13.89</td>
      <td>0.0</td>
      <td>0.550</td>
      <td>6.642</td>
      <td>85.1</td>
      <td>3.4211</td>
      <td>5.0</td>
      <td>276.0</td>
      <td>16.4</td>
      <td>392.78</td>
      <td>9.69</td>
      <td>28.7</td>
    </tr>
    <tr>
      <th>181</th>
      <td>0.06888</td>
      <td>0.0</td>
      <td>2.46</td>
      <td>0.0</td>
      <td>0.488</td>
      <td>6.144</td>
      <td>62.2</td>
      <td>2.5979</td>
      <td>3.0</td>
      <td>193.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.45</td>
      <td>36.2</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.11504</td>
      <td>0.0</td>
      <td>2.89</td>
      <td>0.0</td>
      <td>0.445</td>
      <td>6.163</td>
      <td>69.6</td>
      <td>3.4952</td>
      <td>2.0</td>
      <td>276.0</td>
      <td>18.0</td>
      <td>391.83</td>
      <td>11.34</td>
      <td>21.4</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.677
    Model:                            OLS   Adj. R-squared:                  0.666
    Method:                 Least Squares   F-statistic:                     59.55
    Date:                Tue, 16 Mar 2021   Prob (F-statistic):           4.00e-76
    Time:                        08:54:04   Log-Likelihood:                -1097.9
    No. Observations:                 354   AIC:                             2222.
    Df Residuals:                     341   BIC:                             2272.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     20.5884      6.768      3.042      0.003       7.275      33.901
    CRIM          -0.1295      0.057     -2.289      0.023      -0.241      -0.018
    ZN             0.0374      0.018      2.062      0.040       0.002       0.073
    INDUS          0.0217      0.088      0.247      0.805      -0.151       0.194
    CHAS           4.3746      1.204      3.632      0.000       2.006       6.744
    NOX          -24.8474      5.554     -4.474      0.000     -35.772     -13.923
    RM             6.2752      0.491     12.777      0.000       5.309       7.241
    AGE           -0.0515      0.017     -3.099      0.002      -0.084      -0.019
    DIS           -1.5562      0.262     -5.938      0.000      -2.072      -1.041
    RAD            0.2687      0.092      2.909      0.004       0.087       0.450
    TAX           -0.0117      0.005     -2.243      0.026      -0.022      -0.001
    PTRATIO       -0.9224      0.185     -4.980      0.000      -1.287      -0.558
    B              0.0132      0.004      3.547      0.000       0.006       0.020
    ==============================================================================
    Omnibus:                      196.846   Durbin-Watson:                   2.012
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1890.185
    Skew:                           2.145   Prob(JB):                         0.00
    Kurtosis:                      13.476   Cond. No.                     1.50e+04
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

    r2 score: 0.6866075961248881
    mse: 22.49635161757549
    rmse: 4.7430319013870745
    mae: 3.4213775802266184


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.716
    Model:                            OLS   Adj. R-squared:                  0.708
    Method:                 Least Squares   F-statistic:                     86.66
    Date:                Tue, 16 Mar 2021   Prob (F-statistic):           1.36e-87
    Time:                        08:54:04   Log-Likelihood:                 32.938
    No. Observations:                 354   AIC:                            -43.88
    Df Residuals:                     343   BIC:                            -1.313
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.1380      0.275     11.422      0.000       2.598       3.678
    CRIM           -0.0129      0.002     -5.618      0.000      -0.017      -0.008
    CHAS            0.1670      0.049      3.405      0.001       0.071       0.263
    NOX            -1.0493      0.216     -4.857      0.000      -1.474      -0.624
    RM              0.2266      0.020     11.521      0.000       0.188       0.265
    DIS            -0.0499      0.010     -5.251      0.000      -0.069      -0.031
    RAD             0.0105      0.004      2.891      0.004       0.003       0.018
    TAX            -0.0004      0.000     -2.284      0.023      -0.001   -5.91e-05
    PTRATIO        -0.0406      0.007     -5.779      0.000      -0.054      -0.027
    B               0.0006      0.000      4.189      0.000       0.000       0.001
    pow(AGE, 2) -2.138e-05   5.76e-06     -3.710      0.000   -3.27e-05      -1e-05
    ==============================================================================
    Omnibus:                      112.758   Durbin-Watson:                   1.858
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              834.780
    Skew:                           1.118   Prob(JB):                    5.37e-182
    Kurtosis:                      10.183   Cond. No.                     1.71e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.71e+05. This might indicate that there are
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

    r2 score: 0.7533049338189626
    mse: 17.708594345321988
    rmse: 4.208158070382098
    mae: 3.0014876257221323



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
