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
    Dep. Variable:                      y   R-squared:                       0.724
    Model:                            OLS   Adj. R-squared:                  0.713
    Method:                 Least Squares   F-statistic:                     68.59
    Date:                Mon, 23 Nov 2020   Prob (F-statistic):           1.14e-86
    Time:                        22:16:23   Log-Likelihood:                -1067.4
    No. Observations:                 354   AIC:                             2163.
    Df Residuals:                     340   BIC:                             2217.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         38.9656      6.588      5.915      0.000      26.007      51.924
    CRIM          -0.1365      0.038     -3.607      0.000      -0.211      -0.062
    ZN             0.0598      0.018      3.375      0.001       0.025       0.095
    INDUS          0.0528      0.073      0.721      0.471      -0.091       0.197
    CHAS           3.2390      1.142      2.835      0.005       0.992       5.486
    NOX          -18.9859      4.761     -3.988      0.000     -28.351      -9.621
    RM             3.5202      0.536      6.569      0.000       2.466       4.574
    AGE            0.0001      0.017      0.009      0.993      -0.033       0.033
    DIS           -1.5380      0.251     -6.125      0.000      -2.032      -1.044
    RAD            0.3309      0.078      4.229      0.000       0.177       0.485
    TAX           -0.0112      0.004     -2.560      0.011      -0.020      -0.003
    PTRATIO       -0.9968      0.164     -6.075      0.000      -1.320      -0.674
    B              0.0104      0.003      3.025      0.003       0.004       0.017
    LSTAT         -0.5512      0.063     -8.705      0.000      -0.676      -0.427
    ==============================================================================
    Omnibus:                      117.165   Durbin-Watson:                   1.576
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              446.191
    Skew:                           1.415   Prob(JB):                     1.29e-97
    Kurtosis:                       7.716   Cond. No.                     1.55e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.55e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
res.pvalues
```




    const      8.110143e-09
    CRIM       3.557312e-04
    ZN         8.243261e-04
    INDUS      4.712342e-01
    CHAS       4.852990e-03
    NOX        8.169866e-05
    RM         1.905718e-10
    AGE        9.930348e-01
    DIS        2.510370e-09
    RAD        3.023042e-05
    TAX        1.091144e-02
    PTRATIO    3.328726e-09
    B          2.678575e-03
    LSTAT      1.400212e-16
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
      <th>306</th>
      <td>0.07503</td>
      <td>33.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.472</td>
      <td>7.420</td>
      <td>71.9</td>
      <td>3.0992</td>
      <td>7.0</td>
      <td>222.0</td>
      <td>18.4</td>
      <td>396.90</td>
      <td>6.47</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>218</th>
      <td>0.11069</td>
      <td>0.0</td>
      <td>13.89</td>
      <td>1.0</td>
      <td>0.550</td>
      <td>5.951</td>
      <td>93.8</td>
      <td>2.8893</td>
      <td>5.0</td>
      <td>276.0</td>
      <td>16.4</td>
      <td>396.90</td>
      <td>17.92</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>249</th>
      <td>0.19073</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.431</td>
      <td>6.718</td>
      <td>17.5</td>
      <td>7.8265</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>393.74</td>
      <td>6.56</td>
      <td>26.2</td>
    </tr>
    <tr>
      <th>329</th>
      <td>0.06724</td>
      <td>0.0</td>
      <td>3.24</td>
      <td>0.0</td>
      <td>0.460</td>
      <td>6.333</td>
      <td>17.2</td>
      <td>5.2146</td>
      <td>4.0</td>
      <td>430.0</td>
      <td>16.9</td>
      <td>375.21</td>
      <td>7.34</td>
      <td>22.6</td>
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
    Dep. Variable:                  PRICE   R-squared:                       0.651
    Model:                            OLS   Adj. R-squared:                  0.641
    Method:                 Least Squares   F-statistic:                     63.91
    Date:                Mon, 23 Nov 2020   Prob (F-statistic):           3.11e-72
    Time:                        22:16:23   Log-Likelihood:                -1109.0
    No. Observations:                 354   AIC:                             2240.
    Df Residuals:                     343   BIC:                             2283.
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     18.3219      6.940      2.640      0.009       4.671      31.973
    CRIM          -0.1831      0.042     -4.364      0.000      -0.266      -0.101
    ZN             0.0429      0.019      2.215      0.027       0.005       0.081
    INDUS         -0.1000      0.074     -1.359      0.175      -0.245       0.045
    CHAS           4.3620      1.269      3.438      0.001       1.867       6.857
    NOX          -25.1844      5.044     -4.993      0.000     -35.105     -15.264
    RM             6.0514      0.505     11.982      0.000       5.058       7.045
    DIS           -1.2697      0.264     -4.808      0.000      -1.789      -0.750
    RAD            0.1511      0.056      2.675      0.008       0.040       0.262
    PTRATIO       -1.1179      0.182     -6.157      0.000      -1.475      -0.761
    B              0.0149      0.004      3.890      0.000       0.007       0.022
    ==============================================================================
    Omnibus:                      174.282   Durbin-Watson:                   1.705
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1328.627
    Skew:                           1.916   Prob(JB):                    3.11e-289
    Kurtosis:                      11.683   Cond. No.                     9.83e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 9.83e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
results = smf.ols('PRICE ~ CRIM + ZN', data=dat).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  PRICE   R-squared:                       0.227
    Model:                            OLS   Adj. R-squared:                  0.222
    Method:                 Least Squares   F-statistic:                     51.47
    Date:                Mon, 23 Nov 2020   Prob (F-statistic):           2.50e-20
    Time:                        22:16:23   Log-Likelihood:                -1249.7
    No. Observations:                 354   AIC:                             2505.
    Df Residuals:                     351   BIC:                             2517.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     22.4342      0.535     41.935      0.000      21.382      23.486
    CRIM          -0.3101      0.050     -6.228      0.000      -0.408      -0.212
    ZN             0.1350      0.020      6.724      0.000       0.096       0.175
    ==============================================================================
    Omnibus:                      110.593   Durbin-Watson:                   1.817
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              265.558
    Skew:                           1.545   Prob(JB):                     2.16e-58
    Kurtosis:                       5.909   Cond. No.                         30.1
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
