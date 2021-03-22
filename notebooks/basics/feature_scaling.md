>**Note**: This is a generated markdown export from the Jupyter notebook file [feature_scaling.ipynb](feature_scaling.ipynb).
>You can also view the notebook with the [nbviewer](https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master/notebooks/basics/feature_scaling.ipynb) from Jupyter. 

# Feature scaling


```python
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn import preprocessing, pipeline, datasets
```


```python
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target_names[iris.target]
```


```python
def plot_data(X, y):
        
    df = pd.DataFrame(X.values, columns=X.columns)
    df['labels'] = y
    
    _ = sns.pairplot(df, hue='labels')
```


```python
X.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data(X, y)
```


    
![png](feature_scaling_files/feature_scaling_5_0.png)
    


## Min / Max


```python
scaler = preprocessing.MinMaxScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.428704</td>
      <td>0.440556</td>
      <td>0.467458</td>
      <td>0.458056</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.230018</td>
      <td>0.181611</td>
      <td>0.299203</td>
      <td>0.317599</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.222222</td>
      <td>0.333333</td>
      <td>0.101695</td>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.416667</td>
      <td>0.416667</td>
      <td>0.567797</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.583333</td>
      <td>0.541667</td>
      <td>0.694915</td>
      <td>0.708333</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data(X_scaled, y)
```


    
![png](feature_scaling_files/feature_scaling_8_0.png)
    


# MaxAbsScaler


```python
scaler = preprocessing.MaxAbsScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.739662</td>
      <td>0.694848</td>
      <td>0.544638</td>
      <td>0.479733</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.104818</td>
      <td>0.099061</td>
      <td>0.255840</td>
      <td>0.304895</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.544304</td>
      <td>0.454545</td>
      <td>0.144928</td>
      <td>0.040000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.645570</td>
      <td>0.636364</td>
      <td>0.231884</td>
      <td>0.120000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.734177</td>
      <td>0.681818</td>
      <td>0.630435</td>
      <td>0.520000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.810127</td>
      <td>0.750000</td>
      <td>0.739130</td>
      <td>0.720000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data(X_scaled, y)
```


    
![png](feature_scaling_files/feature_scaling_11_0.png)
    


## RobustScaler


```python
scaler = preprocessing.RobustScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>1.500000e+02</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.033333</td>
      <td>0.114667</td>
      <td>-1.691429e-01</td>
      <td>-0.067111</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.636974</td>
      <td>0.871733</td>
      <td>5.043709e-01</td>
      <td>0.508158</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.153846</td>
      <td>-2.000000</td>
      <td>-9.571429e-01</td>
      <td>-0.800000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.538462</td>
      <td>-0.400000</td>
      <td>-7.857143e-01</td>
      <td>-0.666667</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.266348e-16</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.461538</td>
      <td>0.600000</td>
      <td>2.142857e-01</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.615385</td>
      <td>2.800000</td>
      <td>7.285714e-01</td>
      <td>0.800000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data(X_scaled, y)
```


    
![png](feature_scaling_files/feature_scaling_14_0.png)
    


## QuantileTransformer (normal)


```python
scaler = preprocessing.QuantileTransformer(output_distribution='normal', n_quantiles=10)

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.029423</td>
      <td>-0.035698</td>
      <td>0.012632</td>
      <td>-0.062428</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.110881</td>
      <td>1.054759</td>
      <td>1.077012</td>
      <td>1.478357</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.199338</td>
      <td>-5.199338</td>
      <td>-5.199338</td>
      <td>-5.199338</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.764710</td>
      <td>-0.764710</td>
      <td>-0.732191</td>
      <td>-0.715053</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.039002</td>
      <td>-0.139710</td>
      <td>0.034842</td>
      <td>-0.139710</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.589456</td>
      <td>0.589456</td>
      <td>0.654452</td>
      <td>0.654452</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.199338</td>
      <td>5.199338</td>
      <td>5.199338</td>
      <td>5.199338</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data(X_scaled, y)
```


    
![png](feature_scaling_files/feature_scaling_17_0.png)
    


## QuantileTransformer (uniform)


```python
scaler = preprocessing.QuantileTransformer(output_distribution='uniform', n_quantiles=10)

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.490118</td>
      <td>0.487531</td>
      <td>0.501493</td>
      <td>0.492239</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.289002</td>
      <td>0.283368</td>
      <td>0.286721</td>
      <td>0.298009</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.222222</td>
      <td>0.222222</td>
      <td>0.232026</td>
      <td>0.237288</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.484444</td>
      <td>0.444444</td>
      <td>0.513889</td>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.722222</td>
      <td>0.722222</td>
      <td>0.743590</td>
      <td>0.743590</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data(X_scaled, y)
```


    
![png](feature_scaling_files/feature_scaling_20_0.png)
    


## PowerTransformer
Apply a power transform featurewise to make data more Gaussian-like.


```python
scaler = preprocessing.PowerTransformer()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.500000e+02</td>
      <td>1.500000e+02</td>
      <td>1.500000e+02</td>
      <td>1.500000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-5.447494e-15</td>
      <td>6.300146e-15</td>
      <td>2.842171e-16</td>
      <td>1.089499e-15</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.003350e+00</td>
      <td>1.003350e+00</td>
      <td>1.003350e+00</td>
      <td>1.003350e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.137770e+00</td>
      <td>-2.759144e+00</td>
      <td>-1.545592e+00</td>
      <td>-1.476845e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-8.956896e-01</td>
      <td>-5.614702e-01</td>
      <td>-1.224374e+00</td>
      <td>-1.189599e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.642955e-02</td>
      <td>-8.191725e-02</td>
      <td>3.225908e-01</td>
      <td>1.596788e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.222371e-01</td>
      <td>5.958605e-01</td>
      <td>7.598052e-01</td>
      <td>7.964903e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.176957e+00</td>
      <td>2.743175e+00</td>
      <td>1.828818e+00</td>
      <td>1.658549e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data(X_scaled, y)
```


    
![png](feature_scaling_files/feature_scaling_23_0.png)
    


## Normalize samples individually to unit norm
Scale input vectors individually to unit norm (vector length).


```python
scaler = mm = pipeline.make_pipeline(preprocessing.MinMaxScaler(), preprocessing.Normalizer())

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.409314</td>
      <td>0.518059</td>
      <td>0.430511</td>
      <td>0.413522</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.123306</td>
      <td>0.302146</td>
      <td>0.219147</td>
      <td>0.233686</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.334888</td>
      <td>0.296698</td>
      <td>0.167761</td>
      <td>0.124490</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.427741</td>
      <td>0.374264</td>
      <td>0.544644</td>
      <td>0.520427</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.491244</td>
      <td>0.914248</td>
      <td>0.594792</td>
      <td>0.595049</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.612540</td>
      <td>0.999174</td>
      <td>0.708205</td>
      <td>0.738046</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data(X_scaled, y)
```


    
![png](feature_scaling_files/feature_scaling_26_0.png)
    


## Standardization
Standardize features by removing the mean and scaling to unit variance


```python
scaler = preprocessing.StandardScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.500000e+02</td>
      <td>1.500000e+02</td>
      <td>1.500000e+02</td>
      <td>1.500000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-1.468455e-15</td>
      <td>-1.823726e-15</td>
      <td>-1.610564e-15</td>
      <td>-9.473903e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.003350e+00</td>
      <td>1.003350e+00</td>
      <td>1.003350e+00</td>
      <td>1.003350e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.870024e+00</td>
      <td>-2.433947e+00</td>
      <td>-1.567576e+00</td>
      <td>-1.447076e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-9.006812e-01</td>
      <td>-5.923730e-01</td>
      <td>-1.226552e+00</td>
      <td>-1.183812e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-5.250608e-02</td>
      <td>-1.319795e-01</td>
      <td>3.364776e-01</td>
      <td>1.325097e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.745011e-01</td>
      <td>5.586108e-01</td>
      <td>7.627583e-01</td>
      <td>7.906707e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.492019e+00</td>
      <td>3.090775e+00</td>
      <td>1.785832e+00</td>
      <td>1.712096e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data(X_scaled, y)
```


    
![png](feature_scaling_files/feature_scaling_29_0.png)
    
