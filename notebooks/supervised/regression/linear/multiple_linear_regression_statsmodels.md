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
    Model:                            OLS   Adj. R-squared (uncentered):              0.960
    Method:                 Least Squares   F-statistic:                              648.0
    Date:                Mon, 22 Mar 2021   Prob (F-statistic):                   2.29e-231
    Time:                        15:34:05   Log-Likelihood:                         -1053.8
    No. Observations:                 354   AIC:                                      2134.
    Df Residuals:                     341   BIC:                                      2184.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0999      0.035     -2.878      0.004      -0.168      -0.032
    ZN             0.0592      0.018      3.331      0.001       0.024       0.094
    INDUS         -0.0199      0.072     -0.278      0.781      -0.161       0.121
    CHAS           3.3101      1.042      3.175      0.002       1.260       5.360
    NOX           -1.2175      3.972     -0.306      0.759      -9.031       6.596
    RM             5.5832      0.376     14.866      0.000       4.845       6.322
    AGE           -0.0027      0.017     -0.165      0.869      -0.035       0.030
    DIS           -0.9872      0.233     -4.230      0.000      -1.446      -0.528
    RAD            0.2094      0.079      2.659      0.008       0.055       0.364
    TAX           -0.0103      0.005     -2.230      0.026      -0.019      -0.001
    PTRATIO       -0.3311      0.130     -2.539      0.012      -0.587      -0.075
    B              0.0159      0.003      5.232      0.000       0.010       0.022
    LSTAT         -0.4554      0.062     -7.335      0.000      -0.578      -0.333
    ==============================================================================
    Omnibus:                      104.991   Durbin-Watson:                   1.922
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              561.569
    Skew:                           1.132   Prob(JB):                    1.14e-122
    Kurtosis:                       8.740   Cond. No.                     8.67e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.67e+03. This might indicate that there are
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

    r2 score: 0.6795316906832454
    mse: 28.37866666236672
    rmse: 5.327163097030794
    mae: 3.4989287892909133


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.753
    Model:                            OLS   Adj. R-squared:                  0.744
    Method:                 Least Squares   F-statistic:                     79.81
    Date:                Mon, 22 Mar 2021   Prob (F-statistic):           7.67e-95
    Time:                        15:34:05   Log-Likelihood:                -1035.6
    No. Observations:                 354   AIC:                             2099.
    Df Residuals:                     340   BIC:                             2153.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         36.5330      6.033      6.055      0.000      24.665      48.401
    CRIM          -0.1158      0.033     -3.496      0.001      -0.181      -0.051
    ZN             0.0635      0.017      3.751      0.000       0.030       0.097
    INDUS          0.0046      0.068      0.067      0.946      -0.130       0.139
    CHAS           3.1559      0.992      3.181      0.002       1.204       5.107
    NOX          -16.8304      4.575     -3.678      0.000     -25.830      -7.831
    RM             3.4761      0.499      6.969      0.000       2.495       4.457
    AGE            0.0057      0.016      0.360      0.719      -0.025       0.037
    DIS           -1.5715      0.242     -6.491      0.000      -2.048      -1.095
    RAD            0.3510      0.079      4.470      0.000       0.197       0.505
    TAX           -0.0136      0.004     -3.067      0.002      -0.022      -0.005
    PTRATIO       -0.8692      0.153     -5.696      0.000      -1.169      -0.569
    B              0.0098      0.003      3.216      0.001       0.004       0.016
    LSTAT         -0.5503      0.061     -9.004      0.000      -0.671      -0.430
    ==============================================================================
    Omnibus:                      112.205   Durbin-Watson:                   1.949
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              442.104
    Skew:                           1.335   Prob(JB):                     9.96e-97
    Kurtosis:                       7.779   Cond. No.                     1.56e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.56e+04. This might indicate that there are
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

    r2 score: 0.7045971686745646
    mse: 26.159024894464316
    rmse: 5.11458941602005
    mae: 3.5415151625490706


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
      <th>222</th>
      <td>0.62356</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>1.0</td>
      <td>0.507</td>
      <td>6.879</td>
      <td>77.7</td>
      <td>3.2721</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>390.39</td>
      <td>9.93</td>
      <td>27.5</td>
    </tr>
    <tr>
      <th>51</th>
      <td>0.04337</td>
      <td>21.0</td>
      <td>5.64</td>
      <td>0.0</td>
      <td>0.439</td>
      <td>6.115</td>
      <td>63.0</td>
      <td>6.8147</td>
      <td>4.0</td>
      <td>243.0</td>
      <td>16.8</td>
      <td>393.97</td>
      <td>9.43</td>
      <td>20.5</td>
    </tr>
    <tr>
      <th>192</th>
      <td>0.08664</td>
      <td>45.0</td>
      <td>3.44</td>
      <td>0.0</td>
      <td>0.437</td>
      <td>7.178</td>
      <td>26.3</td>
      <td>6.4798</td>
      <td>5.0</td>
      <td>398.0</td>
      <td>15.2</td>
      <td>390.49</td>
      <td>2.87</td>
      <td>36.4</td>
    </tr>
    <tr>
      <th>419</th>
      <td>11.81230</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.718</td>
      <td>6.824</td>
      <td>76.5</td>
      <td>1.7940</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>48.45</td>
      <td>22.74</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0.01381</td>
      <td>80.0</td>
      <td>0.46</td>
      <td>0.0</td>
      <td>0.422</td>
      <td>7.875</td>
      <td>32.0</td>
      <td>5.6484</td>
      <td>4.0</td>
      <td>255.0</td>
      <td>14.4</td>
      <td>394.23</td>
      <td>2.97</td>
      <td>50.0</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.694
    Model:                            OLS   Adj. R-squared:                  0.684
    Method:                 Least Squares   F-statistic:                     64.55
    Date:                Mon, 22 Mar 2021   Prob (F-statistic):           3.68e-80
    Time:                        15:34:05   Log-Likelihood:                -1073.5
    No. Observations:                 354   AIC:                             2173.
    Df Residuals:                     341   BIC:                             2223.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     22.6052      6.480      3.488      0.001       9.859      35.352
    CRIM          -0.1596      0.036     -4.382      0.000      -0.231      -0.088
    ZN             0.0483      0.019      2.580      0.010       0.011       0.085
    INDUS         -0.0214      0.076     -0.282      0.778      -0.171       0.128
    CHAS           3.8280      1.099      3.482      0.001       1.666       5.990
    NOX          -22.6330      5.034     -4.496      0.000     -32.534     -12.732
    RM             5.9195      0.465     12.728      0.000       5.005       6.834
    AGE           -0.0381      0.017     -2.272      0.024      -0.071      -0.005
    DIS           -1.6514      0.269     -6.143      0.000      -2.180      -1.123
    RAD            0.3092      0.087      3.550      0.000       0.138       0.480
    TAX           -0.0131      0.005     -2.665      0.008      -0.023      -0.003
    PTRATIO       -0.9982      0.169     -5.912      0.000      -1.330      -0.666
    B              0.0135      0.003      4.023      0.000       0.007       0.020
    ==============================================================================
    Omnibus:                      163.954   Durbin-Watson:                   1.987
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1109.319
    Skew:                           1.819   Prob(JB):                    1.30e-241
    Kurtosis:                      10.872   Cond. No.                     1.54e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.54e+04. This might indicate that there are
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

    r2 score: 0.6562678678062692
    mse: 30.438765135521837
    rmse: 5.517133779012599
    mae: 3.6334454537891943


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.717
    Model:                            OLS   Adj. R-squared:                  0.709
    Method:                 Least Squares   F-statistic:                     86.97
    Date:                Mon, 22 Mar 2021   Prob (F-statistic):           8.85e-88
    Time:                        15:34:05   Log-Likelihood:                 33.946
    No. Observations:                 354   AIC:                            -45.89
    Df Residuals:                     343   BIC:                            -3.329
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.2082      0.284     11.301      0.000       2.650       3.767
    CRIM           -0.0124      0.002     -7.852      0.000      -0.016      -0.009
    CHAS            0.1390      0.048      2.910      0.004       0.045       0.233
    NOX            -1.0347      0.214     -4.828      0.000      -1.456      -0.613
    RM              0.2253      0.020     11.457      0.000       0.187       0.264
    DIS            -0.0479      0.010     -4.741      0.000      -0.068      -0.028
    RAD             0.0120      0.004      3.262      0.001       0.005       0.019
    TAX            -0.0005      0.000     -2.753      0.006      -0.001      -0.000
    PTRATIO        -0.0455      0.007     -6.578      0.000      -0.059      -0.032
    B               0.0006      0.000      4.295      0.000       0.000       0.001
    pow(AGE, 2) -1.558e-05   6.12e-06     -2.548      0.011   -2.76e-05   -3.55e-06
    ==============================================================================
    Omnibus:                       70.444   Durbin-Watson:                   2.082
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              448.760
    Skew:                           0.635   Prob(JB):                     3.57e-98
    Kurtosis:                       8.367   Cond. No.                     1.82e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.82e+05. This might indicate that there are
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

    r2 score: 0.7147895942818092
    mse: 25.25644750886994
    rmse: 5.025579320722134
    mae: 3.2413842869550886



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```


    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_0.png)
    
