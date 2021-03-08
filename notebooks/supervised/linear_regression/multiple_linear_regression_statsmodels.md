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
    Dep. Variable:                      y   R-squared:                       0.751
    Model:                            OLS   Adj. R-squared:                  0.742
    Method:                 Least Squares   F-statistic:                     79.00
    Date:                Fri, 19 Feb 2021   Prob (F-statistic):           2.79e-94
    Time:                        08:35:50   Log-Likelihood:                -1048.9
    No. Observations:                 354   AIC:                             2126.
    Df Residuals:                     340   BIC:                             2180.
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         29.7377      6.100      4.875      0.000      17.739      41.736
    CRIM          -0.1137      0.036     -3.131      0.002      -0.185      -0.042
    ZN             0.0542      0.017      3.219      0.001       0.021       0.087
    INDUS          0.0726      0.072      1.004      0.316      -0.070       0.215
    CHAS           3.0485      0.999      3.051      0.002       1.083       5.014
    NOX          -14.2167      4.732     -3.004      0.003     -23.524      -4.909
    RM             4.4143      0.515      8.570      0.000       3.401       5.427
    AGE           -0.0122      0.016     -0.757      0.450      -0.044       0.019
    DIS           -1.3907      0.236     -5.904      0.000      -1.854      -0.927
    RAD            0.3033      0.077      3.924      0.000       0.151       0.455
    TAX           -0.0134      0.004     -3.036      0.003      -0.022      -0.005
    PTRATIO       -0.8636      0.156     -5.520      0.000      -1.171      -0.556
    B              0.0076      0.003      2.380      0.018       0.001       0.014
    LSTAT         -0.5039      0.068     -7.461      0.000      -0.637      -0.371
    ==============================================================================
    Omnibus:                      125.123   Durbin-Watson:                   2.053
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              470.196
    Skew:                           1.528   Prob(JB):                    7.91e-103
    Kurtosis:                       7.747   Cond. No.                     1.52e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.52e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
res.pvalues
```




    const      1.672514e-06
    CRIM       1.892172e-03
    ZN         1.407950e-03
    INDUS      3.159285e-01
    CHAS       2.463375e-03
    NOX        2.857998e-03
    RM         3.685387e-16
    AGE        4.495515e-01
    DIS        8.594426e-09
    RAD        1.052647e-04
    TAX        2.579078e-03
    PTRATIO    6.731455e-08
    B          1.784170e-02
    LSTAT      7.247243e-13
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
      <th>265</th>
      <td>0.76162</td>
      <td>20.0</td>
      <td>3.97</td>
      <td>0.0</td>
      <td>0.6470</td>
      <td>5.560</td>
      <td>62.8</td>
      <td>1.9865</td>
      <td>5.0</td>
      <td>264.0</td>
      <td>13.0</td>
      <td>392.40</td>
      <td>10.45</td>
      <td>22.8</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0.03510</td>
      <td>95.0</td>
      <td>2.68</td>
      <td>0.0</td>
      <td>0.4161</td>
      <td>7.853</td>
      <td>33.2</td>
      <td>5.1180</td>
      <td>4.0</td>
      <td>224.0</td>
      <td>14.7</td>
      <td>392.78</td>
      <td>3.81</td>
      <td>48.5</td>
    </tr>
    <tr>
      <th>453</th>
      <td>8.24809</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.7130</td>
      <td>7.393</td>
      <td>99.3</td>
      <td>2.4527</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>375.87</td>
      <td>16.74</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>241</th>
      <td>0.10612</td>
      <td>30.0</td>
      <td>4.93</td>
      <td>0.0</td>
      <td>0.4280</td>
      <td>6.095</td>
      <td>65.1</td>
      <td>6.3361</td>
      <td>6.0</td>
      <td>300.0</td>
      <td>16.6</td>
      <td>394.62</td>
      <td>12.40</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>247</th>
      <td>0.19657</td>
      <td>22.0</td>
      <td>5.86</td>
      <td>0.0</td>
      <td>0.4310</td>
      <td>6.226</td>
      <td>79.2</td>
      <td>8.0555</td>
      <td>7.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>376.14</td>
      <td>10.15</td>
      <td>20.5</td>
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
    Dep. Variable:                  PRICE   R-squared:                       0.694
    Model:                            OLS   Adj. R-squared:                  0.685
    Method:                 Least Squares   F-statistic:                     77.94
    Date:                Fri, 19 Feb 2021   Prob (F-statistic):           4.54e-82
    Time:                        08:35:50   Log-Likelihood:                -1085.3
    No. Observations:                 354   AIC:                             2193.
    Df Residuals:                     343   BIC:                             2235.
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     13.9125      6.331      2.198      0.029       1.460      26.365
    CRIM          -0.1766      0.039     -4.511      0.000      -0.254      -0.100
    ZN             0.0419      0.018      2.299      0.022       0.006       0.078
    INDUS         -0.0917      0.071     -1.290      0.198      -0.231       0.048
    CHAS           3.7712      1.090      3.461      0.001       1.628       5.914
    NOX          -24.2610      4.903     -4.948      0.000     -33.905     -14.617
    RM             6.5192      0.468     13.931      0.000       5.599       7.440
    DIS           -1.1558      0.249     -4.643      0.000      -1.645      -0.666
    RAD            0.1305      0.054      2.428      0.016       0.025       0.236
    PTRATIO       -1.0161      0.171     -5.956      0.000      -1.352      -0.681
    B              0.0110      0.004      3.120      0.002       0.004       0.018
    ==============================================================================
    Omnibus:                      152.901   Durbin-Watson:                   2.106
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              892.545
    Skew:                           1.726   Prob(JB):                    1.54e-194
    Kurtosis:                       9.971   Cond. No.                     9.69e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 9.69e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
results = smf.ols('PRICE ~ CRIM + ZN', data=dat).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  PRICE   R-squared:                       0.249
    Model:                            OLS   Adj. R-squared:                  0.245
    Method:                 Least Squares   F-statistic:                     58.28
    Date:                Fri, 19 Feb 2021   Prob (F-statistic):           1.40e-22
    Time:                        08:35:50   Log-Likelihood:                -1244.4
    No. Observations:                 354   AIC:                             2495.
    Df Residuals:                     351   BIC:                             2506.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     22.4927      0.534     42.119      0.000      21.442      23.543
    CRIM          -0.3361      0.048     -6.981      0.000      -0.431      -0.241
    ZN             0.1287      0.019      6.698      0.000       0.091       0.167
    ==============================================================================
    Omnibus:                      119.051   Durbin-Watson:                   1.908
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              303.853
    Skew:                           1.637   Prob(JB):                     1.04e-66
    Kurtosis:                       6.143   Cond. No.                         31.7
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
