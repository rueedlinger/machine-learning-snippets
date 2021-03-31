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
    Dep. Variable:                      y   R-squared (uncentered):                   0.959
    Model:                            OLS   Adj. R-squared (uncentered):              0.957
    Method:                 Least Squares   F-statistic:                              612.9
    Date:                Wed, 31 Mar 2021   Prob (F-statistic):                   2.05e-227
    Time:                        17:17:25   Log-Likelihood:                         -1074.0
    No. Observations:                 354   AIC:                                      2174.
    Df Residuals:                     341   BIC:                                      2224.
    Df Model:                          13                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0792      0.040     -1.978      0.049      -0.158      -0.000
    ZN             0.0582      0.018      3.266      0.001       0.023       0.093
    INDUS         -0.0231      0.086     -0.267      0.789      -0.193       0.146
    CHAS           4.4007      1.175      3.746      0.000       2.090       6.711
    NOX           -4.9257      4.138     -1.190      0.235     -13.065       3.214
    RM             6.3484      0.394     16.131      0.000       5.574       7.123
    AGE           -0.0122      0.017     -0.722      0.471      -0.046       0.021
    DIS           -1.0851      0.236     -4.600      0.000      -1.549      -0.621
    RAD            0.1422      0.087      1.641      0.102      -0.028       0.313
    TAX           -0.0081      0.005     -1.540      0.125      -0.018       0.002
    PTRATIO       -0.4915      0.137     -3.594      0.000      -0.760      -0.223
    B              0.0146      0.003      4.366      0.000       0.008       0.021
    LSTAT         -0.3352      0.062     -5.416      0.000      -0.457      -0.213
    ==============================================================================
    Omnibus:                      108.804   Durbin-Watson:                   1.976
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              597.920
    Skew:                           1.170   Prob(JB):                    1.46e-130
    Kurtosis:                       8.921   Cond. No.                     8.47e+03
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 8.47e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")
```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_9_1.png)
    



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

    r2 score: 0.627897787617306
    mse: 22.96234619319922
    rmse: 4.791904234560539
    mae: 3.1400087657668627


### Full model with an intercept



```python
model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.760
    Model:                            OLS   Adj. R-squared:                  0.751
    Method:                 Least Squares   F-statistic:                     83.00
    Date:                Wed, 31 Mar 2021   Prob (F-statistic):           5.21e-97
    Time:                        17:17:25   Log-Likelihood:                -1053.3
    No. Observations:                 354   AIC:                             2135.
    Df Residuals:                     340   BIC:                             2189.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         40.8443      6.283      6.501      0.000      28.486      53.203
    CRIM          -0.1028      0.038     -2.706      0.007      -0.178      -0.028
    ZN             0.0586      0.017      3.486      0.001       0.026       0.092
    INDUS          0.0186      0.082      0.227      0.820      -0.142       0.179
    CHAS           4.1838      1.110      3.769      0.000       2.000       6.367
    NOX          -21.7234      4.685     -4.636      0.000     -30.940     -12.507
    RM             4.0033      0.518      7.729      0.000       2.984       5.022
    AGE           -0.0054      0.016     -0.337      0.737      -0.037       0.026
    DIS           -1.7078      0.242     -7.043      0.000      -2.185      -1.231
    RAD            0.3193      0.086      3.701      0.000       0.150       0.489
    TAX           -0.0125      0.005     -2.491      0.013      -0.022      -0.003
    PTRATIO       -1.1169      0.161     -6.935      0.000      -1.434      -0.800
    B              0.0093      0.003      2.864      0.004       0.003       0.016
    LSTAT         -0.4622      0.062     -7.500      0.000      -0.583      -0.341
    ==============================================================================
    Omnibus:                      100.725   Durbin-Watson:                   1.949
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              357.607
    Skew:                           1.225   Prob(JB):                     2.22e-78
    Kurtosis:                       7.271   Cond. No.                     1.52e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.52e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")

```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_14_1.png)
    



```python
predicted = result.predict(sm.add_constant(X_test))

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.6418385887269948
    mse: 22.102062403856987
    rmse: 4.701283059320827
    mae: 3.335409951225634


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
      <th>305</th>
      <td>0.05479</td>
      <td>33.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.472</td>
      <td>6.616</td>
      <td>58.1</td>
      <td>3.3700</td>
      <td>7.0</td>
      <td>222.0</td>
      <td>18.4</td>
      <td>393.36</td>
      <td>8.93</td>
      <td>28.4</td>
    </tr>
    <tr>
      <th>475</th>
      <td>6.39312</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.584</td>
      <td>6.162</td>
      <td>97.4</td>
      <td>2.2060</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>302.76</td>
      <td>24.10</td>
      <td>13.3</td>
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
    <tr>
      <th>371</th>
      <td>9.23230</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.631</td>
      <td>6.216</td>
      <td>100.0</td>
      <td>1.1691</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>366.15</td>
      <td>9.53</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>317</th>
      <td>0.24522</td>
      <td>0.0</td>
      <td>9.90</td>
      <td>0.0</td>
      <td>0.544</td>
      <td>5.782</td>
      <td>71.7</td>
      <td>4.0317</td>
      <td>4.0</td>
      <td>304.0</td>
      <td>18.4</td>
      <td>396.90</td>
      <td>15.94</td>
      <td>19.8</td>
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
    Dep. Variable:                   MEDV   R-squared:                       0.721
    Model:                            OLS   Adj. R-squared:                  0.711
    Method:                 Least Squares   F-statistic:                     73.35
    Date:                Wed, 31 Mar 2021   Prob (F-statistic):           8.90e-87
    Time:                        17:17:26   Log-Likelihood:                -1080.4
    No. Observations:                 354   AIC:                             2187.
    Df Residuals:                     341   BIC:                             2237.
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     25.9001      6.423      4.032      0.000      13.266      38.535
    CRIM          -0.1327      0.041     -3.256      0.001      -0.213      -0.053
    ZN             0.0465      0.018      2.575      0.010       0.011       0.082
    INDUS         -0.0190      0.088     -0.216      0.829      -0.192       0.154
    CHAS           4.9907      1.191      4.191      0.000       2.648       7.333
    NOX          -24.7576      5.032     -4.920      0.000     -34.655     -14.860
    RM             6.2863      0.452     13.915      0.000       5.398       7.175
    AGE           -0.0473      0.016     -2.924      0.004      -0.079      -0.015
    DIS           -1.7659      0.261     -6.759      0.000      -2.280      -1.252
    RAD            0.2637      0.093      2.845      0.005       0.081       0.446
    TAX           -0.0117      0.005     -2.154      0.032      -0.022      -0.001
    PTRATIO       -1.1411      0.174     -6.574      0.000      -1.483      -0.800
    B              0.0113      0.004      3.235      0.001       0.004       0.018
    ==============================================================================
    Omnibus:                      133.722   Durbin-Watson:                   1.922
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              765.631
    Skew:                           1.476   Prob(JB):                    5.56e-167
    Kurtosis:                       9.572   Cond. No.                     1.48e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.48e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
_ = sm.qqplot(result.resid, fit=True, line="s")
```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_20_1.png)
    



```python
predicted = result.predict(X_test)

print("r2 score: {}".format(metrics.r2_score(y_test, predicted)))
print("mse: {}".format(metrics.mean_squared_error(y_test, predicted)))
print("rmse: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
print("mae: {}".format(metrics.mean_absolute_error(y_test, predicted)))
```

    r2 score: 0.5222200901364318
    mse: 29.483693806043647
    rmse: 5.4298889312806065
    mae: 3.6303938697486355


### Model with a polynomial and the target variable log transformed


```python
result = smf.ols('np.log(MEDV) ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + pow(AGE, 2)', data=dat).fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           np.log(MEDV)   R-squared:                       0.714
    Model:                            OLS   Adj. R-squared:                  0.706
    Method:                 Least Squares   F-statistic:                     85.70
    Date:                Wed, 31 Mar 2021   Prob (F-statistic):           5.31e-87
    Time:                        17:17:26   Log-Likelihood:                 26.069
    No. Observations:                 354   AIC:                            -30.14
    Df Residuals:                     343   BIC:                             12.42
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept       3.4043      0.280     12.139      0.000       2.853       3.956
    CRIM           -0.0104      0.002     -5.840      0.000      -0.014      -0.007
    CHAS            0.1868      0.052      3.605      0.000       0.085       0.289
    NOX            -1.1396      0.212     -5.369      0.000      -1.557      -0.722
    RM              0.2228      0.019     11.712      0.000       0.185       0.260
    DIS            -0.0540      0.010     -5.380      0.000      -0.074      -0.034
    RAD             0.0097      0.004      2.499      0.013       0.002       0.017
    TAX            -0.0005      0.000     -2.189      0.029      -0.001   -4.65e-05
    PTRATIO        -0.0491      0.007     -6.931      0.000      -0.063      -0.035
    B               0.0006      0.000      3.707      0.000       0.000       0.001
    pow(AGE, 2) -1.992e-05   5.91e-06     -3.370      0.001   -3.15e-05   -8.29e-06
    ==============================================================================
    Omnibus:                       53.522   Durbin-Watson:                   1.972
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              364.178
    Skew:                           0.357   Prob(JB):                     8.31e-80
    Kurtosis:                       7.917   Cond. No.                     1.73e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.73e+05. This might indicate that there are
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

    r2 score: 0.5988174141288799
    mse: 24.756889684872984
    rmse: 4.975629576734283
    mae: 3.2356123011415683



```python
_ = sm.qqplot(result.resid, fit=True, line="q")
```

    /Users/mru/.local/share/virtualenvs/machine-learning-snippets-mLikUPnf/lib/python3.8/site-packages/statsmodels/graphics/gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
      ax.plot(x, y, fmt, **plot_style)



    
![png](multiple_linear_regression_statsmodels_files/multiple_linear_regression_statsmodels_27_1.png)
    
