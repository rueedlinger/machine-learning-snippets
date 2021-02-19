>**Note**: This is a generated markdown export from the Jupyter notebook file [multiple_linear_regression_sklearn.ipynb](multiple_linear_regression_sklearn.ipynb).

## Multiple Linear Regression with scikit-learn


```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn import linear_model, datasets, metrics, model_selection, feature_selection, preprocessing

from scipy import stats

```


```python
boston = datasets.load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

```


```python
print('shape:', X.shape)
```

    shape: (506, 13)



```python
X.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.displot(y)
```




    <seaborn.axisgrid.FacetGrid at 0x106b8cd30>




    
![png](multiple_linear_regression_sklearn_files/multiple_linear_regression_sklearn_5_1.png)
    



```python
sns.pairplot(X);
```




    <seaborn.axisgrid.PairGrid at 0x124f43b80>




    
![png](multiple_linear_regression_sklearn_files/multiple_linear_regression_sklearn_6_1.png)
    



```python
f, axes = plt.subplots(2, 1)

sns.histplot(y, ax=axes[0])
sns.boxplot(data=y, orient='h', ax=axes[1])


```




    <AxesSubplot:>




    
![png](multiple_linear_regression_sklearn_files/multiple_linear_regression_sklearn_7_1.png)
    



```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7)

print('train samples:', len(X_train))
print('test samples', len(X_test))

```

    train samples: 354
    test samples 152



```python
#import warnings
#warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
```


```python
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
```




    LinearRegression()




```python
print('No coef:', len(lr.coef_))
print('Coefficients: 
', lr.coef_)
```

    No coef: 13
    Coefficients: 
     [-1.07691620e-01  5.74394184e-02 -2.97562761e-02  2.30846578e+00
     -1.49122875e+01  2.92922610e+00 -1.66954492e-03 -1.59796261e+00
      2.92993371e-01 -1.40254984e-02 -9.63610083e-01  8.19098264e-03
     -5.32276201e-01]



```python
predicted = lr.predict(X_test)
```


```python
fig, ax = plt.subplots()
ax.scatter(y_test, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], ls='--', color='red')
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
```




    Text(0, 0.5, 'Predicted')




    
![png](multiple_linear_regression_sklearn_files/multiple_linear_regression_sklearn_13_1.png)
    



```python
residual = (y_test - predicted)
```


```python
fig, ax = plt.subplots()
ax.scatter(y_test, residual)
plt.axhline(0, color='red', ls='--')
ax.set_xlabel('y')
ax.set_ylabel('residual')
```




    Text(0, 0.5, 'residual')




    
![png](multiple_linear_regression_sklearn_files/multiple_linear_regression_sklearn_15_1.png)
    



```python
sns.displot(residual, kind="kde");
```




    <seaborn.axisgrid.FacetGrid at 0x129a99a60>




    
![png](multiple_linear_regression_sklearn_files/multiple_linear_regression_sklearn_16_1.png)
    


The trainig scores


```python
metrics.r2_score(y_train, lr.predict(X_train))
```




    0.7572133181038798




```python
metrics.mean_squared_error(y_train, lr.predict(X_train))
```




    19.253887940700547




```python
metrics.r2_score(y_test, predicted)
```




    0.6874348612215382




```python
metrics.mean_squared_error(y_test, predicted)
```




    29.416888376117484




```python
print(lr.intercept_)
print(lr.coef_)
```

    42.944698484449205
    [-1.07691620e-01  5.74394184e-02 -2.97562761e-02  2.30846578e+00
     -1.49122875e+01  2.92922610e+00 -1.66954492e-03 -1.59796261e+00
      2.92993371e-01 -1.40254984e-02 -9.63610083e-01  8.19098264e-03
     -5.32276201e-01]
