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
    Dep. Variable:                      y   R-squared (uncentered):                   0.956
    Model:                            OLS   Adj. R-squared (uncentered):              0.954
    Method:                 Least Squares   F-statistic:                              565.0
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):                   1.18e-221
    Time:                        10:31:59   Log-Likelihood:                         -1083.1
    No. Observations:                 354   AIC:                                      2192.
    Df Residuals:                     341   BIC:                                      2242.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.1216      0.039     -3.116      0.002      -0.198      -0.045
    ZN             0.0530      0.018      3.004      0.003       0.018       0.088
    INDUS          0.0176      0.078      0.225      0.822      -0.136       0.171
    CHAS           2.6012      1.130      2.302      0.022       0.379       4.824
    NOX           -2.5152      4.491     -0.560      0.576     -11.349       6.318
    RM             5.7568      0.381     15.108      0.000       5.007       6.506
    AGE            0.0045      0.018      0.256      0.798      -0.030       0.039
    DIS           -0.8472      0.239     -3.543      0.000      -1.318      -0.377
    RAD            0.2009      0.084      2.386      0.018       0.035       0.367
    TAX           -0.0093      0.005     -1.911      0.057      -0.019       0.000
    PTRATIO       -0.3718      0.136     -2.733      0.007      -0.639      -0.104
    B              0.0131      0.004      3.472      0.001       0.006       0.020
    LSTAT         -0.4512      0.063     -7.213      0.000      -0.574      -0.328
    ==============================================================================
    Omnibus:                      138.168   Durbin-Watson:                   2.150
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              868.369
    Skew:                           1.499   Prob(JB):                    2.73e-189
    Kurtosis:                      10.063   Cond. No.                     9.04e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 9.04e+03. This might indicate that there are
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

    r2 score: 0.7693573244595222
    mse: 19.135254302527482
    rmse: 4.374386162940748
    mae: 3.273712718466068


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.720
    Model:                            OLS   Adj. R-squared:                  0.709
    Method:                 Least Squares   F-statistic:                     67.27
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           1.19e-85
    Time:                        10:31:59   Log-Likelihood:                -1063.2
    No. Observations:                 354   AIC:                             2154.
    Df Residuals:                     340   BIC:                             2209.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         39.2071      6.176      6.348      0.000      27.059      51.356
    CRIM          -0.1290      0.037     -3.488      0.001      -0.202      -0.056
    ZN             0.0515      0.017      3.087      0.002       0.019       0.084
    INDUS          0.0143      0.074      0.194      0.847      -0.131       0.159
    CHAS           2.2285      1.072      2.080      0.038       0.121       4.336
    NOX          -17.7781      4.885     -3.639      0.000     -27.387      -8.169
    RM             3.5047      0.506      6.926      0.000       2.509       4.500
    AGE            0.0088      0.017      0.523      0.601      -0.024       0.042
    DIS           -1.4724      0.247     -5.963      0.000      -1.958      -0.987
    RAD            0.3446      0.083      4.157      0.000       0.182       0.508
    TAX           -0.0129      0.005     -2.780      0.006      -0.022      -0.004
    PTRATIO       -0.9697      0.160     -6.077      0.000      -1.284      -0.656
    B              0.0079      0.004      2.176      0.030       0.001       0.015
    LSTAT         -0.5559      0.061     -9.041      0.000      -0.677      -0.435
    ==============================================================================
    Omnibus:                      123.176   Durbin-Watson:                   2.174
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              503.044
    Skew:                           1.467   Prob(JB):                    5.82e-110
    Kurtosis:                       8.049   Cond. No.                     1.46e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.46e+04. This might indicate that there are
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

    r2 score: 0.7837973764259132
    mse: 17.937236347389295
    rmse: 4.235237460566916
    mae: 3.205648472760021


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
      <th>119</th>
      <td>0.14476</td>
      <td>0.0</td>
      <td>10.01</td>
      <td>0.0</td>
      <td>0.547</td>
      <td>5.731</td>
      <td>65.2</td>
      <td>2.7592</td>
      <td>6.0</td>
      <td>432.0</td>
      <td>17.8</td>
      <td>391.50</td>
      <td>13.61</td>
      <td>19.3</td>
    </tr>
    <tr>
      <th>121</th>
      <td>0.07165</td>
      <td>0.0</td>
      <td>25.65</td>
      <td>0.0</td>
      <td>0.581</td>
      <td>6.004</td>
      <td>84.1</td>
      <td>2.1974</td>
      <td>2.0</td>
      <td>188.0</td>
      <td>19.1</td>
      <td>377.67</td>
      <td>14.27</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>464</th>
      <td>7.83932</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.655</td>
      <td>6.209</td>
      <td>65.4</td>
      <td>2.9634</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>13.22</td>
      <td>21.4</td>
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
    <tr>
      <th>439</th>
      <td>9.39063</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>5.627</td>
      <td>93.9</td>
      <td>1.8172</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>22.88</td>
      <td>12.8</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.653
    Model:                            OLS   Adj. R-squared:                  0.641
    Method:                 Least Squares   F-statistic:                     53.42
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           7.49e-71
    Time:                        10:31:59   Log-Likelihood:                -1101.4
    No. Observations:                 354   AIC:                             2229.
    Df Residuals:                     341   BIC:                             2279.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     24.2214      6.617      3.661      0.000      11.207      37.236
    CRIM          -0.1766      0.041     -4.340      0.000      -0.257      -0.097
    ZN             0.0449      0.019      2.422      0.016       0.008       0.081
    INDUS         -0.0500      0.082     -0.612      0.541      -0.211       0.111
    CHAS           2.6340      1.191      2.212      0.028       0.292       4.976
    NOX          -20.6986      5.421     -3.818      0.000     -31.362     -10.036
    RM             5.8692      0.482     12.183      0.000       4.922       6.817
    AGE           -0.0446      0.017     -2.557      0.011      -0.079      -0.010
    DIS           -1.6309      0.274     -5.955      0.000      -2.170      -1.092
    RAD            0.3150      0.092      3.420      0.001       0.134       0.496
    TAX           -0.0131      0.005     -2.537      0.012      -0.023      -0.003
    PTRATIO       -0.9973      0.177     -5.621      0.000      -1.346      -0.648
    B              0.0101      0.004      2.491      0.013       0.002       0.018
    ==============================================================================
    Omnibus:                      202.365   Durbin-Watson:                   2.208
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1866.088
    Skew:                           2.244   Prob(JB):                         0.00
    Kurtosis:                      13.314   Cond. No.                     1.44e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.44e+04. This might indicate that there are
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

    r2 score: 0.7506386567689014
    mse: 20.688247327886625
    rmse: 4.548433502634355
    mae: 3.3279061486233776


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.672
    Model:                            OLS   Adj. R-squared:                  0.663
    Method:                 Least Squares   F-statistic:                     70.39
    Date:                Fri, 12 Mar 2021   Prob (F-statistic):           6.13e-77
    Time:                        10:31:59   Log-Likelihood:                 18.313
    No. Observations:                 354   AIC:                            -14.63
    Df Residuals:                     343   BIC:                             27.94
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.3073      0.279     11.875      0.000       2.760       3.855
    CRIM           -0.0125      0.002     -7.272      0.000      -0.016      -0.009
    CHAS            0.1044      0.050      2.084      0.038       0.006       0.203
    NOX            -0.9210      0.224     -4.118      0.000      -1.361      -0.481
    RM              0.2169      0.020     11.057      0.000       0.178       0.256
    DIS            -0.0459      0.010     -4.611      0.000      -0.066      -0.026
    RAD             0.0139      0.004      3.676      0.000       0.006       0.021
    TAX            -0.0007      0.000     -3.305      0.001      -0.001      -0.000
    PTRATIO        -0.0429      0.007     -6.016      0.000      -0.057      -0.029
    B               0.0004      0.000      2.117      0.035    2.56e-05       0.001
    pow(AGE, 2) -1.858e-05   6.05e-06     -3.071      0.002   -3.05e-05   -6.68e-06
    ==============================================================================
    Omnibus:                      103.912   Durbin-Watson:                   2.177
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              716.680
    Skew:                           1.032   Prob(JB):                    2.37e-156
    Kurtosis:                       9.658   Cond. No.                     1.68e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.68e+05. This might indicate that there are
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

    r2 score: 0.770519965705847
    mse: 19.038795848518916
    rmse: 4.36334686319102
    mae: 2.9948556968361455



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
