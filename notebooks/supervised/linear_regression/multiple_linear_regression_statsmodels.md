>**Note**: This is a generated markdown export from the Jupyter notebook file [multiple_linear_regression_statsmodels.ipynb](multiple_linear_regression_statsmodels.ipynb).

## Multiple Linear Regression with statsmodels


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

from sklearn import datasets, model_selection

```


```python
boston = datasets.load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

```


```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7)
```


```python
model = sm.OLS(y_train, sm.add_constant(X_train))
res = model.fit()

print(res.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.728
    Model:                            OLS   Adj. R-squared:                  0.718
    Method:                 Least Squares   F-statistic:                     70.06
    Date:                Fri, 30 Oct 2020   Prob (F-statistic):           8.50e-88
    Time:                        09:04:53   Log-Likelihood:                -1053.8
    No. Observations:                 354   AIC:                             2136.
    Df Residuals:                     340   BIC:                             2190.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         44.0878      6.177      7.137      0.000      31.938      56.238
    CRIM          -0.1248      0.037     -3.354      0.001      -0.198      -0.052
    ZN             0.0665      0.017      3.807      0.000       0.032       0.101
    INDUS         -0.0580      0.075     -0.769      0.443      -0.206       0.090
    CHAS           3.1769      1.181      2.689      0.008       0.853       5.501
    NOX          -17.8546      4.636     -3.851      0.000     -26.974      -8.735
    RM             3.0239      0.501      6.040      0.000       2.039       4.009
    AGE           -0.0066      0.016     -0.416      0.678      -0.038       0.025
    DIS           -1.8322      0.247     -7.410      0.000      -2.319      -1.346
    RAD            0.3892      0.082      4.743      0.000       0.228       0.551
    TAX           -0.0151      0.005     -3.155      0.002      -0.025      -0.006
    PTRATIO       -0.9300      0.158     -5.897      0.000      -1.240      -0.620
    B              0.0084      0.003      2.577      0.010       0.002       0.015
    LSTAT         -0.5012      0.061     -8.284      0.000      -0.620      -0.382
    ==============================================================================
    Omnibus:                      122.408   Durbin-Watson:                   1.802
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              503.572
    Skew:                           1.454   Prob(JB):                    4.47e-110
    Kurtosis:                       8.068   Cond. No.                     1.50e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.5e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
res.pvalues
```




    const      5.793213e-12
    CRIM       8.869828e-04
    ZN         1.666095e-04
    INDUS      4.426706e-01
    CHAS       7.514282e-03
    NOX        1.405692e-04
    RM         4.045495e-09
    AGE        6.779796e-01
    DIS        1.009368e-12
    RAD        3.110157e-06
    TAX        1.750243e-03
    PTRATIO    8.928530e-09
    B          1.037824e-02
    LSTAT      2.781755e-15
    dtype: float64




```python
res.pvalues < 0.05
```




    const       True
    CRIM        True
    ZN          True
    INDUS      False
    CHAS        True
    NOX         True
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
dat = X_train.copy()
dat['PRICE'] = y_train
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
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>406</th>
      <td>20.71620</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.659</td>
      <td>4.138</td>
      <td>100.0</td>
      <td>1.1781</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>370.22</td>
      <td>23.34</td>
      <td>11.9</td>
    </tr>
    <tr>
      <th>257</th>
      <td>0.61154</td>
      <td>20.0</td>
      <td>3.97</td>
      <td>0.0</td>
      <td>0.647</td>
      <td>8.704</td>
      <td>86.9</td>
      <td>1.8010</td>
      <td>5.0</td>
      <td>264.0</td>
      <td>13.0</td>
      <td>389.70</td>
      <td>5.12</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>219</th>
      <td>0.11425</td>
      <td>0.0</td>
      <td>13.89</td>
      <td>1.0</td>
      <td>0.550</td>
      <td>6.373</td>
      <td>92.4</td>
      <td>3.3633</td>
      <td>5.0</td>
      <td>276.0</td>
      <td>16.4</td>
      <td>393.74</td>
      <td>10.50</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>169</th>
      <td>2.44953</td>
      <td>0.0</td>
      <td>19.58</td>
      <td>0.0</td>
      <td>0.605</td>
      <td>6.402</td>
      <td>95.2</td>
      <td>2.2625</td>
      <td>5.0</td>
      <td>403.0</td>
      <td>14.7</td>
      <td>330.04</td>
      <td>11.32</td>
      <td>22.3</td>
    </tr>
    <tr>
      <th>132</th>
      <td>0.59005</td>
      <td>0.0</td>
      <td>21.89</td>
      <td>0.0</td>
      <td>0.624</td>
      <td>6.372</td>
      <td>97.9</td>
      <td>2.3274</td>
      <td>4.0</td>
      <td>437.0</td>
      <td>21.2</td>
      <td>385.76</td>
      <td>11.12</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
results = smf.ols('PRICE ~ CRIM + ZN + INDUS + CHAS + NOX + RM + DIS + RAD + PTRATIO + B', data=dat).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  PRICE   R-squared:                       0.658
    Model:                            OLS   Adj. R-squared:                  0.648
    Method:                 Least Squares   F-statistic:                     66.05
    Date:                Fri, 30 Oct 2020   Prob (F-statistic):           8.00e-74
    Time:                        09:04:58   Log-Likelihood:                -1094.4
    No. Observations:                 354   AIC:                             2211.
    Df Residuals:                     343   BIC:                             2253.
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     26.6504      6.555      4.065      0.000      13.757      39.544
    CRIM          -0.1832      0.041     -4.489      0.000      -0.264      -0.103
    ZN             0.0441      0.019      2.362      0.019       0.007       0.081
    INDUS         -0.2310      0.075     -3.069      0.002      -0.379      -0.083
    CHAS           4.1816      1.308      3.196      0.002       1.609       6.755
    NOX          -26.7490      4.873     -5.489      0.000     -36.334     -17.164
    RM             5.2540      0.477     11.008      0.000       4.315       6.193
    DIS           -1.5998      0.264     -6.068      0.000      -2.118      -1.081
    RAD            0.2174      0.054      4.057      0.000       0.112       0.323
    PTRATIO       -1.1178      0.174     -6.442      0.000      -1.459      -0.776
    B              0.0139      0.004      3.916      0.000       0.007       0.021
    ==============================================================================
    Omnibus:                      160.668   Durbin-Watson:                   1.875
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1107.863
    Skew:                           1.766   Prob(JB):                    2.69e-241
    Kurtosis:                      10.914   Cond. No.                     9.62e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 9.62e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
results = smf.ols('PRICE ~ CRIM + ZN', data=dat).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  PRICE   R-squared:                       0.237
    Model:                            OLS   Adj. R-squared:                  0.232
    Method:                 Least Squares   F-statistic:                     54.39
    Date:                Fri, 30 Oct 2020   Prob (F-statistic):           2.66e-21
    Time:                        09:04:59   Log-Likelihood:                -1236.6
    No. Observations:                 354   AIC:                             2479.
    Df Residuals:                     351   BIC:                             2491.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     22.0958      0.517     42.717      0.000      21.078      23.113
    CRIM          -0.2970      0.049     -6.068      0.000      -0.393      -0.201
    ZN             0.1313      0.018      7.194      0.000       0.095       0.167
    ==============================================================================
    Omnibus:                      111.809   Durbin-Watson:                   2.054
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              280.122
    Skew:                           1.537   Prob(JB):                     1.49e-61
    Kurtosis:                       6.089   Cond. No.                         32.2
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
